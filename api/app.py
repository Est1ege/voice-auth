from fastapi import FastAPI, UploadFile, File, Form, Depends, HTTPException, Query, WebSocket, WebSocketDisconnect, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import APIKeyHeader
from fastapi.responses import FileResponse, JSONResponse, HTMLResponse
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from datetime import datetime 
import os
import tempfile
import zipfile
import uuid
import json
from motor.motor_asyncio import AsyncIOMotorClient
import numpy as np
import logging
import httpx
import shutil
import subprocess
from pathlib import Path
import asyncio
import logging
import requests
from bson import ObjectId


app = FastAPI(title="Голосовая аутентификация API", version="1.0.0")

# CORS middleware для веб-интерфейса
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # В продакшене заменить на список разрешенных доменов
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("/app/logs/api.log"),
    ]
)
logger = logging.getLogger("api")



# Менеджер соединений WebSocket
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def broadcast(self, message: dict):
        for connection in self.active_connections:
            await connection.send_json(message)

manager = ConnectionManager()

# Конфигурация
MODEL_PATH = os.environ.get("MODEL_PATH", "/shared/models")
DB_URL = os.environ.get("DATABASE_URL", "mongodb://db:27017/voice_auth")
AUDIO_PATH = os.environ.get("AUDIO_PATH", "/shared/audio")
UPLOAD_FOLDER = os.environ.get("UPLOAD_FOLDER", "/shared/temp")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
LOG_FOLDER = "/shared/logs/test_results"



# URL ML сервиса из переменных окружения
ML_SERVICE_URL = os.environ.get("ML_SERVICE_URL", "http://ml_model:5000")



# Создание директории для аудио, если не существует
os.makedirs(AUDIO_PATH, exist_ok=True)
os.makedirs("/app/logs", exist_ok=True)

# Подключение к базе данных
client = None
db = None

# Модели данных
class User(BaseModel):
    id: Optional[str] = None
    name: str
    email: str
    role: str = "user"
    active: bool = False
    created_at: Optional[datetime] = None
    voice_samples: Optional[List[str]] = []
    voice_embeddings: Optional[List[List[float]]] = []
    # Дополнительные поля для КПП
    department: Optional[str] = None
    position: Optional[str] = None
    access_level: Optional[str] = "standard"
    has_photo: Optional[bool] = False

class VoiceSample(BaseModel):
    id: str
    user_id: str
    file_path: str
    embedding: Optional[List[float]] = None
    created_at: datetime = Field(default_factory=datetime.now)

class LogEntry(BaseModel):
    id: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.now)
    event_type: str
    user_id: Optional[str] = None
    user_name: Optional[str] = None
    success: bool
    details: Optional[dict] = None
    ip_address: Optional[str] = None

class SystemStatus(BaseModel):
    api_status: str
    ml_status: str
    db_status: str
    users_count: int
    active_users_count: int

# Подключение к сервисам
async def startup_db_client():
    global client, db
    try:
        client = AsyncIOMotorClient(DB_URL)
        db = client.voice_auth
        # Тестовое обращение для проверки соединения
        await db.command("ping")
        logger.info("Connected to MongoDB")
    except Exception as e:
        logger.error(f"Failed to connect to MongoDB: {e}")
        client = None
        db = None

async def shutdown_db_client():
    global client
    if client:
        client.close()
        logger.info("Disconnected from MongoDB")

app.add_event_handler("startup", startup_db_client)
app.add_event_handler("shutdown", shutdown_db_client)

# Зависимость для проверки подключения к БД
async def get_db():
    global db
    if db is None:
        await startup_db_client()
        if db is None:
            raise HTTPException(status_code=500, detail="Database connection failed")
    return db

# ML сервис
# В api/app.py

async def call_ml_service(method, endpoint=None, json=None):
    """Вызов методов ML-сервиса с полной обратной совместимостью"""
    ml_service_url = os.environ.get("ML_SERVICE_URL", "http://ml_model:5000")

    # Определяем, используется старый или новый формат вызова
    if endpoint is None:
        # Старый формат: call_ml_service("endpoint")
        endpoint = method
        method = "POST"
        data_payload = None
    elif isinstance(endpoint, dict) and json is None:
        # Старый формат: call_ml_service("endpoint", data_dict)
        data_payload = endpoint
        endpoint = method
        method = "POST"
    elif isinstance(method, dict):
        # Случай: call_ml_service(data_dict, "endpoint") - неправильный порядок
        logger.warning("Неправильный порядок аргументов в call_ml_service!")
        data_payload = method
        endpoint = endpoint if isinstance(endpoint, str) else "/extract_embedding"
        method = "POST"
    else:
        # Новый формат: call_ml_service("POST", "/endpoint", data)
        data_payload = json

    # Убедимся, что endpoint - это строка
    if not isinstance(endpoint, str):
        # Если endpoint не строка, используем значение по умолчанию
        logger.warning(f"Endpoint должен быть строкой, получено: {type(endpoint)}")
        endpoint = "/extract_embedding"  # Значение по умолчанию

    # Убедимся, что endpoint начинается с "/"
    if not endpoint.startswith("/"):
        endpoint = f"/{endpoint}"

    url = f"{ml_service_url}{endpoint}"

    logger.info(f"Calling ML service: {method} {url}")

    # Проверка и коррекция пути к аудиофайлу для совместимости с ML-сервисом
    if data_payload and 'audio_path' in data_payload:
        original_path = data_payload['audio_path']

        # Если путь в /tmp, перенаправляем в /shared/temp
        if original_path.startswith('/tmp/'):
            filename = os.path.basename(original_path)
            shared_path = f"/shared/temp/{filename}"

            # Проверяем, существует ли файл в /shared/temp
            if os.path.exists(shared_path):
                logger.info(f"Redirecting audio path from {original_path} to {shared_path}")
                data_payload['audio_path'] = shared_path
            else:
                # Ищем по имени файла в общей директории
                shared_dir = "/shared/temp"
                if os.path.exists(shared_dir):
                    for file in os.listdir(shared_dir):
                        if file.endswith('.wav'):
                            logger.info(f"Found potential match in shared directory: {file}")
                            shared_path = f"{shared_dir}/{file}"
                            logger.info(f"Using shared file instead: {shared_path}")
                            data_payload['audio_path'] = shared_path
                            break

        # Проверяем и логируем существование файла перед отправкой запроса
        if os.path.exists(data_payload['audio_path']):
            logger.info(f"File confirmed to exist before ML call: {data_payload['audio_path']}")
            logger.info(f"File size: {os.path.getsize(data_payload['audio_path'])} bytes")
        else:
            logger.warning(f"File does not exist before ML call: {data_payload['audio_path']}")

            # Ищем файлы в shared/temp в качестве запасного варианта
            shared_dir = "/shared/temp"
            if os.path.exists(shared_dir):
                temp_files = os.listdir(shared_dir)
                if temp_files:
                    # Используем первый доступный WAV файл
                    for file in temp_files:
                        if file.endswith('.wav'):
                            replacement_path = f"{shared_dir}/{file}"
                            logger.info(f"Using replacement file: {replacement_path}")
                            data_payload['audio_path'] = replacement_path
                            break

    async with httpx.AsyncClient() as client:
        try:
            if method.upper() == "GET":
                response = await client.get(url, timeout=30.0)
            elif method.upper() == "POST":
                response = await client.post(url, json=data_payload, timeout=30.0)
            else:
                raise ValueError(f"Unsupported HTTP method: {method}")

            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error calling ML service ({method} {url}): {e}")
            raise HTTPException(status_code=e.response.status_code, detail=f"ML service error: {e}")
        except httpx.RequestError as e:
            logger.error(f"Request error calling ML service ({method} {url}): {e}")
            # Возвращаем заглушку для endpoint training status
            if "training" in endpoint and "status" in endpoint:
                return {
                    "task_id": endpoint.split("/")[-2],
                    "status": "unknown",
                    "progress": 0,
                    "message": f"Cannot connect to ML service: {e}",
                    "type": None
                }
            raise HTTPException(status_code=500, detail=f"Cannot connect to ML service: {e}")
        except Exception as e:
            logger.error(f"Unexpected error calling ML service ({method} {url}): {e}")
            raise HTTPException(status_code=500, detail=f"Error communicating with ML service: {str(e)}")

# Аудио сервис
async def call_audio_processor(endpoint, data=None, file=None):
    async with httpx.AsyncClient() as client:
        try:
            if file:
                files = {"audio": (file.filename, await file.read())}
                response = await client.post(
                    f"http://audio_processor:5000/{endpoint}",
                    files=files,
                    timeout=30.0
                )
            else:
                response = await client.post(
                    f"http://audio_processor:5000/{endpoint}",
                    json=data,
                    timeout=30.0
                )
            return response.json()
        except Exception as e:
            logger.error(f"Failed to call audio processor: {e}")
            raise HTTPException(status_code=500, detail=f"Audio processor error: {str(e)}")

@app.get("/health")
async def health_check():
    """
    Проверка состояния API сервиса.
    Должен возвращать 200 OK, если сервис работает нормально.
    """
    try:
        # Проверка подключения к базе данных
        db = await get_db()
        return {
            "status": "ok",
            "message": "API сервис работает",
            "database_connected": db is not None,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=500, detail=f"Сервис не работает: {str(e)}")

@app.get("/api/system/status")
async def system_status(db=Depends(get_db)):
    try:
        # Проверка статуса ML сервиса
        try:
            ml_status = "ok"
            await call_ml_service("health", {})
        except:
            ml_status = "error"
        
        # Подсчет пользователей
        users_count = await db.users.count_documents({})
        active_users_count = await db.users.count_documents({"active": True})
        
        # Определение устройства вычислений
        # (это может быть заменено на реальное определение устройства, если необходимо)
        device = "CPU"  # или "GPU" если используется GPU
        
        return {
            "api_status": "ok",
            "ml_status": ml_status,
            "db_status": "ok" if db else "error",
            "users_count": users_count,
            "active_users_count": active_users_count,
            "device": device  # Добавляем поле device
        }
    except Exception as e:
        logger.error(f"Error getting system status: {e}")
        # Вместо вызова HTTPException возвращаем объект с ошибкой
        # чтобы не возвращать 500 статус
        return {
            "api_status": "error",
            "ml_status": "unknown",
            "db_status": "unknown",
            "users_count": 0,
            "active_users_count": 0,
            "device": "Unknown",
            "error": str(e)
        }

@app.get("/api/users")
async def get_users(db=Depends(get_db)):
    try:
        users = await db.users.find().to_list(1000)
        # Преобразуем ObjectId в строки
        for user in users:
            user["id"] = str(user.pop("_id"))
        return {"users": users}
    except Exception as e:
        logger.error(f"Error getting users: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/users", status_code=201)
async def create_user(user: User, db=Depends(get_db)):
    try:
        user_dict = user.dict(exclude={"id"})
        user_dict["created_at"] = datetime.datetime.now()
        result = await db.users.insert_one(user_dict)
        user_id = str(result.inserted_id)
        
        # Создаем директорию для образцов голоса пользователя
        os.makedirs(os.path.join(AUDIO_PATH, user_id), exist_ok=True)
        
        # Логирование
        log_entry = LogEntry(
            event_type="user_created",
            user_id=user_id,
            user_name=user.name,
            success=True,
            details={"email": user.email, "role": user.role}
        )
        await db.logs.insert_one(log_entry.dict(exclude={"id"}))
        
        return {"user_id": user_id, "success": True}
    except Exception as e:
        logger.error(f"Error creating user: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/users/{user_id}")
async def get_user(user_id: str, db=Depends(get_db)):
    from bson.objectid import ObjectId
    try:
        user = await db.users.find_one({"_id": ObjectId(user_id)})
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
        
        user["id"] = str(user.pop("_id"))
        return {"user": user}
    except Exception as e:
        logger.error(f"Error getting user: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# api/app.py - добавьте эндпоинт для загрузки фото пользователя
@app.post("/api/users/{user_id}/photo")
async def upload_user_photo(
    user_id: str,
    photo: UploadFile = File(...),
    db=Depends(get_db)
):
    """Загрузка фотографии пользователя"""
    try:
        # Проверка существования пользователя
        from bson.objectid import ObjectId
        user = await db.users.find_one({"_id": ObjectId(user_id)})
        
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
        
        # Создание директории пользователя, если она не существует
        user_dir = os.path.join(AUDIO_PATH, user_id)
        os.makedirs(user_dir, exist_ok=True)
        
        # Путь для сохранения фото
        photo_path = os.path.join(user_dir, "photo.jpg")
        
        # Сохранение фото
        content = await photo.read()
        with open(photo_path, "wb") as f:
            f.write(content)
        
        # Обновление информации о пользователе
        await db.users.update_one(
            {"_id": ObjectId(user_id)},
            {"$set": {"has_photo": True}}
        )
        
        return {"success": True, "message": "Photo uploaded successfully"}
    except Exception as e:
        logger.error(f"Error uploading user photo: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.route('/api/system/reinitialize', methods=['POST'])
def reinitialize_system():
    """Fully reinitialize the system for troubleshooting"""
    try:
        # Hit the ML service reinitialize endpoint
        response = requests.post(f"{API_URL}/ml/system/reinitialize")
        result = response.json()

        if result.get("success", False):
            flash('Система успешно переинициализирована', 'success')
        else:
            flash(f'Ошибка при переинициализации: {result.get("error", "Неизвестная ошибка")}', 'danger')

        return redirect(url_for('system_status'))

    except Exception as e:
        flash(f'Ошибка при переинициализации системы: {str(e)}', 'danger')
        return redirect(url_for('system_status'))

@app.get("/api/users/{user_id}/voice_samples")
async def get_voice_samples(user_id: str, db=Depends(get_db)):
    from bson.objectid import ObjectId
    try:
        # Проверка существования пользователя
        user = await db.users.find_one({"_id": ObjectId(user_id)})
        if not user:
            raise HTTPException(status_code=404, detail="User not found")

        # Получение образцов голоса
        samples = await db.voice_samples.find({"user_id": user_id}).to_list(100)
        for sample in samples:
            sample["id"] = str(sample.pop("_id"))

        return {"samples": samples}
    except Exception as e:
        logger.error(f"Error getting voice samples: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/users/{user_id}/activate")
async def activate_user(user_id: str, db=Depends(get_db)):
    from bson.objectid import ObjectId
    try:
        # Проверка существования пользователя
        user = await db.users.find_one({"_id": ObjectId(user_id)})
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
        
        # Проверка наличия достаточного количества образцов голоса
        samples_count = len(user.get("voice_samples", []))
        if samples_count < 5:
            raise HTTPException(
                status_code=400, 
                detail=f"Insufficient voice samples. Need at least 5, but got {samples_count}"
            )
        
        # Обновление статуса пользователя
        await db.users.update_one(
            {"_id": ObjectId(user_id)},
            {"$set": {"active": True}}
        )
        
        # Обновление модели распознавания
        await call_ml_service("POST", "/update_model", {
            "user_id": user_id,
            "embeddings": user.get("voice_embeddings", [])
        })
        
        # Логирование
        log_entry = LogEntry(
            event_type="user_activated",
            user_id=user_id,
            user_name=user.get("name"),
            success=True
        )
        await db.logs.insert_one(log_entry.dict(exclude={"id"}))
        
        return {"success": True, "message": "User activated successfully"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error activating user: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.websocket("/ws/auth-monitor")
async def websocket_auth_monitor(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            # Ждем сообщения от клиента (для поддержания соединения)
            data = await websocket.receive_text()
            
            # Если клиент запрашивает актуальные данные
            if data == "get_recent":
                # Получение последних событий аутентификации
                events = await db.logs.find({
                    "event_type": {"$in": ["authorization_successful", "authorization_attempt", "spoofing_attempt"]}
                }).sort("timestamp", -1).limit(5).to_list(5)
                
                # Форматирование и отправка
                formatted_events = [...] # Тот же код что и в get_recent_auth_events
                await websocket.send_json({"events": formatted_events})
    except WebSocketDisconnect:
        manager.disconnect(websocket)

# api/app.py - обновите эндпоинт authorize_user для поддержки дополнительных данных
@app.post("/api/authorize")
async def authorize_user(
        audio_data: UploadFile = File(...),
        db=Depends(get_db)
):
    """Авторизация пользователя по голосу с улучшенной обработкой аудиофайлов"""
    try:
        # Создание общей директории для временных файлов, если она не существует
        shared_temp_dir = "/shared/temp"
        os.makedirs(shared_temp_dir, exist_ok=True)

        # Генерация уникального имени файла в общей директории
        file_uuid = str(uuid.uuid4())
        temp_file_path = os.path.join(shared_temp_dir, f"auth_{file_uuid}.wav")

        try:
            # Сохранение аудиофайла с проверкой
            with open(temp_file_path, "wb") as f:
                shutil.copyfileobj(audio_data.file, f)

            logger.info(f"Saved audio to: {temp_file_path}")

            # Проверка существования файла и его размера
            if not os.path.exists(temp_file_path):
                logger.error("Файл не был сохранен")
                return {"authorized": False, "message": "Ошибка при обработке аудиофайла"}

            file_size = os.path.getsize(temp_file_path)
            logger.info(f"File exists: {os.path.exists(temp_file_path)}, size: {file_size} bytes")

            # Если файл слишком мал, возможно возникла ошибка с записью
            if file_size < 1000:  # меньше 1 КБ
                logger.warning(f"Audio file is too small: {file_size} bytes")
                return {"authorized": False, "message": "Записанный аудиофайл слишком короткий"}

            # Преобразование формата аудио при необходимости
            final_file_path = temp_file_path
            try:
                # Используем ffmpeg для принудительного преобразования в формат WAV 16kHz, моно
                # Это решает проблему совместимости с алгоритмом распознавания
                converted_path = os.path.join(shared_temp_dir, f"auth_{file_uuid}_conv.wav")
                import subprocess

                # Запуск процесса конвертации
                subprocess.run([
                    'ffmpeg', '-y', '-i', temp_file_path,
                    '-acodec', 'pcm_s16le', '-ar', '16000', '-ac', '1',
                    converted_path
                ], check=True, stderr=subprocess.PIPE)

                # Проверка успешности конвертации
                if os.path.exists(converted_path) and os.path.getsize(converted_path) > 0:
                    final_file_path = converted_path
                    logger.info(f"Successfully converted audio to compatible WAV format: {final_file_path}")
                else:
                    logger.warning("Conversion failed, using original file")
            except Exception as convert_error:
                logger.warning(f"Error converting audio format: {convert_error}, using original file")

            # Проверка спуфинга с использованием преобразованного файла
            anti_spoof_result = await call_ml_service("POST", "/detect_spoofing", {
                "audio_path": final_file_path
            })

            is_spoof = anti_spoof_result.get("is_spoof", False)
            spoof_probability = anti_spoof_result.get("spoof_probability", 0)

            logger.info(f"Anti-spoofing check result: is_spoof={is_spoof}, probability={spoof_probability}")

            if is_spoof:
                # Логирование попытки спуфинга
                log_entry = LogEntry(
                    event_type="spoofing_attempt",
                    success=False,
                    details={
                        "spoof_probability": spoof_probability,
                        "method": "voice_auth",
                        "audio_path": final_file_path
                    }
                )
                await db.logs.insert_one(log_entry.dict(exclude={"id"}))

                return {
                    "authorized": False,
                    "message": "Обнаружена попытка подделки голоса",
                    "spoofing_detected": True,
                    "spoof_probability": spoof_probability
                }

            # Извлечение эмбеддинга из преобразованного файла
            embedding_result = await call_ml_service("POST", "/extract_embedding", {
                "audio_path": final_file_path
            })

            embedding = embedding_result.get("embedding")

            if not embedding:
                logger.error("Failed to extract embedding from audio")
                return {
                    "authorized": False,
                    "message": "Не удалось извлечь голосовой отпечаток. Пожалуйста, говорите громче и четче."
                }

            # Получение детального сравнения с пользователями
            match_result = await call_ml_service("POST", "/match_user_detailed", {
                "embedding": embedding
            })

            matched_user_id = match_result.get("user_id")
            similarity = match_result.get("similarity", 0)
            detailed_similarity = match_result.get("detailed_similarity", {})

            # Порог сходства для авторизации (можно настроить)
            settings = await db.settings.find_one({"type": "system_settings"})
            similarity_threshold = settings.get("voice_similarity_threshold", 0.75) if settings else 0.75

            logger.info(
                f"Match result: user_id={matched_user_id}, similarity={similarity}, threshold={similarity_threshold}")

            if matched_user_id and similarity >= similarity_threshold:
                # Получение информации о пользователе
                from bson.objectid import ObjectId
                user = await db.users.find_one({"_id": ObjectId(matched_user_id)})

                if not user or not user.get("active", False):
                    # Логирование неудачной попытки
                    log_entry = LogEntry(
                        event_type="authorization_attempt",
                        user_id=matched_user_id if user else None,
                        user_name=user.get("name") if user else None,
                        success=False,
                        details={
                            "reason": "User inactive" if user else "User not found",
                            "similarity": similarity,
                            "method": "voice_auth"
                        }
                    )
                    await db.logs.insert_one(log_entry.dict(exclude={"id"}))

                    message = "Пользователь не активен." if user else "Пользователь не найден."
                    return {"authorized": False, "message": message}

                # Успешная авторизация
                user_id_str = str(user["_id"])

                # Логирование успешной авторизации
                log_entry = LogEntry(
                    event_type="authorization_successful",
                    user_id=user_id_str,
                    user_name=user.get("name"),
                    success=True,
                    details={
                        "similarity": similarity,
                        "method": "voice_auth"
                    }
                )
                await db.logs.insert_one(log_entry.dict(exclude={"id"}))

                # Подготовка данных пользователя для ответа
                user_data = {
                    "id": user_id_str,
                    "name": user.get("name", ""),
                    "email": user.get("email", ""),
                    "role": user.get("role", "user"),
                    # Добавляем дополнительные данные для КПП
                    "department": user.get("department", ""),
                    "position": user.get("position", ""),
                    "access_level": user.get("access_level", "standard")
                }

                # Проверка наличия фото пользователя
                photo_path = os.path.join(AUDIO_PATH, user_id_str, "photo.jpg")
                if os.path.exists(photo_path):
                    user_data["has_photo"] = True
                    user_data["photo_url"] = f"/api/users/{user_id_str}/photo"
                else:
                    user_data["has_photo"] = False

                return {
                    "authorized": True,
                    "user": user_data,
                    "similarity": {
                        "score": similarity,
                        "details": detailed_similarity
                    },
                }
            else:
                # Логирование неудачной попытки
                log_entry = LogEntry(
                    event_type="authorization_attempt",
                    success=False,
                    details={
                        "reason": "No match found or low similarity",
                        "similarity": similarity,
                        "method": "voice_auth",
                        "best_match_id": matched_user_id
                    }
                )
                await db.logs.insert_one(log_entry.dict(exclude={"id"}))

                # Если есть близкое совпадение, указываем имя пользователя
                best_match_name = None
                if matched_user_id:
                    from bson.objectid import ObjectId
                    best_match = await db.users.find_one({"_id": ObjectId(matched_user_id)})
                    if best_match:
                        best_match_name = best_match.get("name")

                return {
                    "authorized": False,
                    "message": "Голос не распознан или недостаточно совпадение. Пожалуйста, попробуйте снова.",
                    "similarity": {
                        "score": similarity,
                        "details": detailed_similarity
                    },
                    "best_match": {
                        "id": matched_user_id,
                        "name": best_match_name,
                        "similarity": similarity
                    } if matched_user_id else None
                }
        finally:
            # Удаление всех временных файлов
            try:
                # Список файлов для удаления
                temp_files = [temp_file_path]

                # Добавляем конвертированный файл, если он был создан
                converted_path = os.path.join(shared_temp_dir, f"auth_{file_uuid}_conv.wav")
                if os.path.exists(converted_path):
                    temp_files.append(converted_path)

                # Удаляем все временные файлы
                for file_path in temp_files:
                    if os.path.exists(file_path):
                        os.remove(file_path)
                        logger.info(f"Removed temporary file: {file_path}")
            except Exception as e:
                logger.warning(f"Failed to remove temporary files: {e}")

    except Exception as e:
        logger.error(f"Error in authorization: {e}", exc_info=True)
        return {
            "authorized": False,
            "message": f"Внутренняя ошибка сервера: {str(e)}",
            "error": True
        }

@app.get("/api/logs")
async def get_logs(
    limit: int = Query(50, ge=1, le=1000),
    skip: int = Query(0, ge=0),
    event_type: Optional[str] = None,
    user_id: Optional[str] = None,
    db=Depends(get_db)
):
    try:
        query = {}
        if event_type:
            query["event_type"] = event_type
        if user_id:
            query["user_id"] = user_id
            
        logs = await db.logs.find(query).sort("timestamp", -1).skip(skip).limit(limit).to_list(limit)
        for log in logs:
            log["id"] = str(log.pop("_id"))
            
        return {"logs": logs}
    except Exception as e:
        logger.error(f"Error getting logs: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/training/start")
async def start_training(training_data: dict, db=Depends(get_db)):
    try:
        # Перенаправление запроса к ML-сервису
        response = await call_ml_service("POST", "/training/start", json=training_data)
        
        # Сохранение задачи тренировки в БД
        task = {
            "task_id": response.get("task_id"),
            "model_type": training_data.get("model_type"),
            "status": response.get("status"),
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "parameters": training_data
        }
        
        await db.training_tasks.insert_one(task)
        
        # Логирование
        log_entry = LogEntry(
            event_type="training_started",
            success=True,
            details={
                "task_id": response.get("task_id"),
                "model_type": training_data.get("model_type")
            }
        )
        await db.logs.insert_one(log_entry.dict(exclude={"id"}))
        
        return response
    except Exception as e:
        logger.error(f"Error starting training: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/training/{task_id}/status")
async def api_training_status(task_id: str):
    try:
        # Перенаправляем запрос к ML-сервису
        response = await call_ml_service("GET", f"/training/{task_id}/status")
        return response
    except Exception as e:
        logger.error(f"Error getting training status: {e}")
        return {
            "task_id": task_id,
            "status": "error",
            "progress": 0,
            "message": f"Ошибка получения статуса: {str(e)}",
            "type": None
        }

@app.get("/api/training/list")
async def get_training_list(db=Depends(get_db)):
    try:
        # Получение списка тренировок из БД
        training_tasks = await db.training_tasks.find().to_list(length=100)
        
        # Преобразование ObjectId в строки
        for task in training_tasks:
            task["_id"] = str(task["_id"])
        
        # Запрос актуального статуса у ML-сервиса для активных задач
        for task in training_tasks:
            if task.get("status") not in ["completed", "failed", "canceled"]:
                try:
                    status_response = await call_ml_service("GET", f"/training/{task['task_id']}/status")
                    task["status"] = status_response.get("status", task.get("status"))
                    task["progress"] = status_response.get("progress", task.get("progress", 0))
                except Exception as e:
                    logger.warning(f"Error getting task status from ML service: {e}")
        
        return {"trainings": training_tasks}
    except Exception as e:
        logger.error(f"Error getting training list: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/training/{task_id}/cancel")
async def cancel_training(task_id: str, db=Depends(get_db)):
    """
    Отмена запущенной тренировки модели.
    """
    try:
        # Отправляем запрос на отмену тренировки к ML-сервису
        response = await call_ml_service("POST", f"/training/{task_id}/cancel")
        
        # Логируем событие отмены тренировки
        log_entry = LogEntry(
            event_type="training_cancelled",
            success=True,
            details={
                "task_id": task_id,
                "cancelled_at": datetime.now().isoformat()
            }
        )
        await db.logs.insert_one(log_entry.dict(exclude={"id"}))
        
        # Обновляем статус тренировки в базе данных
        await db.training_tasks.update_one(
            {"task_id": task_id},
            {"$set": {"status": "cancelled", "updated_at": datetime.now().isoformat()}}
        )
        
        return {
            "success": True,
            "message": "Тренировка отменена",
            "task_id": task_id
        }
    except Exception as e:
        logger.error(f"Error cancelling training: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Ошибка при отмене тренировки: {str(e)}"
        )

# Эндпоинты для экспорта/импорта системы
@app.post("/api/system/import")
async def import_system(
    import_file: UploadFile = File(...),
    db=Depends(get_db)
):
    """
    Импортирует систему из ZIP-архива
    """
    try:
        # Проверка формата файла
        if not import_file.filename.lower().endswith('.zip'):
            raise HTTPException(status_code=400, detail="File must be a ZIP archive")
        
        # Создаем временную директорию для распаковки архива
        temp_dir = tempfile.mkdtemp()
        zip_path = os.path.join(temp_dir, "import.zip")
        
        try:
            # Сохраняем загруженный файл
            content = await import_file.read()
            with open(zip_path, "wb") as f:
                f.write(content)
            
            # Распаковываем архив
            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                zip_ref.extractall(temp_dir)
            
            # 1. Импорт пользователей
            if os.path.exists(os.path.join(temp_dir, "users.json")):
                with open(os.path.join(temp_dir, "users.json"), "r") as f:
                    users_data = json.load(f)
                
                # Очищаем существующую коллекцию пользователей
                await db.users.delete_many({})
                
                # Импортируем пользователей
                for user in users_data:
                    # Конвертируем строки ObjectId обратно в объекты ObjectId
                    from bson.objectid import ObjectId
                    user["_id"] = ObjectId(user["_id"]["$oid"] if isinstance(user["_id"], dict) else user["_id"])
                    await db.users.insert_one(user)
            
            # 2. Импорт образцов голоса
            if os.path.exists(os.path.join(temp_dir, "voice_samples.json")):
                with open(os.path.join(temp_dir, "voice_samples.json"), "r") as f:
                    voice_samples_data = json.load(f)
                
                # Очищаем существующую коллекцию образцов
                await db.voice_samples.delete_many({})
                
                # Импортируем образцы
                for sample in voice_samples_data:
                    # Конвертируем строки ObjectId обратно в объекты ObjectId
                    from bson.objectid import ObjectId
                    sample["_id"] = ObjectId(sample["_id"]["$oid"] if isinstance(sample["_id"], dict) else sample["_id"])
                    
                    # Обновляем путь к файлу (из нового расположения)
                    old_path = sample["file_path"]
                    filename = os.path.basename(old_path)
                    user_id = sample["user_id"]
                    
                    # Создаем новый путь к файлу в текущей системе
                    new_path = os.path.join(AUDIO_PATH, user_id, filename)
                    sample["file_path"] = new_path
                    
                    await db.voice_samples.insert_one(sample)
            
            # 3. Импорт аудиофайлов
            audio_import_dir = os.path.join(temp_dir, "audio")
            if os.path.exists(audio_import_dir):
                # Обрабатываем директорию каждого пользователя
                for user_dir in os.listdir(audio_import_dir):
                    user_id = user_dir
                    source_dir = os.path.join(audio_import_dir, user_dir)
                    
                    if os.path.isdir(source_dir):
                        # Создаем директорию для пользователя если нужно
                        target_dir = os.path.join(AUDIO_PATH, user_id)
                        os.makedirs(target_dir, exist_ok=True)
                        
                        # Копируем все WAV файлы
                        for filename in os.listdir(source_dir):
                            if filename.lower().endswith('.wav'):
                                src_path = os.path.join(source_dir, filename)
                                dst_path = os.path.join(target_dir, filename)
                                shutil.copy2(src_path, dst_path)
            
            # Обновляем все ML модели
            try:
                await call_ml_service("rebuild_models", {})
            except Exception as ml_error:
                logger.warning(f"ML service rebuild warning: {ml_error}")
            
            return {
                "success": True,
                "message": "Система успешно импортирована"
            }
            
        finally:
            # Очистка временных файлов
            shutil.rmtree(temp_dir, ignore_errors=True)
    
    except Exception as e:
        logger.error(f"Error importing system: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# api/app.py - обновите эндпоинт get_voice_sample_audio
@app.get("/api/voice_samples/{sample_id}/audio")
async def get_voice_sample_audio(sample_id: str, db=Depends(get_db)):
    """Получение аудиофайла голосового образца с расширенным логированием"""
    try:
        from bson.objectid import ObjectId
        
        logger.info(f"Request for voice sample audio: {sample_id}")
        
        # Находим запись о голосовом образце
        sample = await db.voice_samples.find_one({"_id": ObjectId(sample_id)})
        
        if not sample:
            logger.error(f"Voice sample not found in database: {sample_id}")
            raise HTTPException(status_code=404, detail="Sample not found")
        
        file_path = sample.get("file_path")
        logger.info(f"Sample found, file path: {file_path}")
        
        if not file_path:
            logger.error(f"File path is missing for sample: {sample_id}")
            raise HTTPException(status_code=404, detail="File path is missing")
        
        # Проверяем существование файла
        if not os.path.exists(file_path):
            logger.error(f"Audio file not found at path: {file_path}")
            
            # Проверим, что директории существуют
            user_id = sample.get("user_id")
            user_dir = os.path.join(AUDIO_PATH, user_id)
            
            if not os.path.exists(user_dir):
                logger.error(f"User directory does not exist: {user_dir}")
                
                # Проверим существование корневой директории
                if not os.path.exists(AUDIO_PATH):
                    logger.error(f"Audio root directory does not exist: {AUDIO_PATH}")
                else:
                    # Выведем список директорий в корневой директории
                    logger.info(f"Directories in audio root: {os.listdir(AUDIO_PATH)}")
            else:
                # Выведем список файлов в директории пользователя
                logger.info(f"Files in user directory: {os.listdir(user_dir)}")
            
            # Проверим альтернативные пути
            filename = os.path.basename(file_path)
            
            alternative_paths = [
                os.path.join(AUDIO_PATH, user_id, filename),
                os.path.join(AUDIO_PATH, str(user_id), filename),
                os.path.join("/app", file_path),  # Иногда пути могут быть относительными к контейнеру
                os.path.join("/shared/audio", user_id, filename)  # Стандартный путь в Docker
            ]
            
            logger.info(f"Checking alternative paths: {alternative_paths}")
            
            found_path = None
            for alt_path in alternative_paths:
                if os.path.exists(alt_path):
                    logger.info(f"Found audio file at alternative path: {alt_path}")
                    found_path = alt_path
                    break
            
            if found_path:
                # Обновим путь в базе данных для будущего использования
                await db.voice_samples.update_one(
                    {"_id": ObjectId(sample_id)},
                    {"$set": {"file_path": found_path}}
                )
                logger.info(f"Updated file path in database to: {found_path}")
                return FileResponse(found_path, media_type="audio/wav")
            else:
                raise HTTPException(status_code=404, detail="Audio file not found")
        
        logger.info(f"Serving audio file: {file_path}")
        return FileResponse(file_path, media_type="audio/wav")
    except Exception as e:
        logger.error(f"Error getting voice sample audio: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Маршруты для упрощенного добавления пользователей и переноса системы

@app.post("/api/simple/users")
async def create_simple_user(name: str = Form(...), db=Depends(get_db)):
    """
    Упрощенное создание пользователя только с ФИО
    """
    try:
        # Создаем базовый пользовательский объект
        user_dict = {
            "name": name,
            "email": f"{name.lower().replace(' ', '.')}@example.com",  # Генерируем временный email
            "role": "user",
            "active": False,
            "created_at": datetime.now(),
            "voice_samples": [],
            "voice_embeddings": []
        }
        
        # Добавляем в базу данных
        result = await db.users.insert_one(user_dict)
        user_id = str(result.inserted_id)
        
        # Создаем директорию для образцов голоса пользователя
        os.makedirs(os.path.join(AUDIO_PATH, user_id), exist_ok=True)
        
        # Логирование
        log_entry = LogEntry(
            event_type="user_created_simple",
            user_id=user_id,
            user_name=name,
            success=True,
            details={"method": "simple_api"}
        )
        await db.logs.insert_one(log_entry.dict(exclude={"id"}))
        
        return {"user_id": user_id, "success": True, "name": name}
    except Exception as e:
        logger.error(f"Error creating simple user: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/simple/upload_voice_batch")
async def upload_voice_batch(
    user_id: str = Form(...),
    audio_files: List[UploadFile] = File(...),
    db=Depends(get_db)
):
    """
    Загрузка нескольких аудиофайлов для пользователя сразу
    """
    from bson.objectid import ObjectId
    try:
        # Проверка существования пользователя
        user = await db.users.find_one({"_id": ObjectId(user_id)})
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
        
        # Убедимся, что директория существует
        user_dir = os.path.join(AUDIO_PATH, user_id)
        os.makedirs(user_dir, exist_ok=True)
        
        # Обрабатываем все файлы
        processed_files = []
        
        for audio_file in audio_files:
            if not audio_file.filename.lower().endswith('.wav'):
                continue
                
            # Генерация уникального имени файла
            sample_id = str(uuid.uuid4())
            filename = f"{sample_id}.wav"
            file_path = os.path.join(user_dir, filename)
            
            # Сохранение аудиофайла
            content = await audio_file.read()
            with open(file_path, "wb") as f:
                f.write(content)
            
            # Сброс позиции чтения файла для повторного использования
            await audio_file.seek(0)
            
            try:
                # Запрос на извлечение эмбеддинга
                embedding_data = await call_ml_service("extract_embedding", {
                    "audio_path": file_path
                })
                
                embedding = embedding_data.get("embedding")
                
                # Создание записи образца голоса
                voice_sample = VoiceSample(
                    id=sample_id,
                    user_id=user_id,
                    file_path=file_path,
                    embedding=embedding
                )
                
                # Добавление образца в базу данных
                await db.voice_samples.insert_one(voice_sample.dict())
                
                # Обновление пользователя
                await db.users.update_one(
                    {"_id": ObjectId(user_id)},
                    {
                        "$push": {
                            "voice_samples": sample_id,
                            "voice_embeddings": embedding
                        }
                    }
                )
                
                processed_files.append(sample_id)
            except Exception as e:
                logger.error(f"Error processing audio file {audio_file.filename}: {e}")
                # Продолжаем с другими файлами даже при ошибке
        
        # Получение обновленного списка образцов
        updated_user = await db.users.find_one({"_id": ObjectId(user_id)})
        sample_count = len(updated_user.get("voice_samples", []))
        
        # Логирование
        log_entry = LogEntry(
            event_type="voice_samples_batch_upload",
            user_id=user_id,
            user_name=user.get("name"),
            success=True,
            details={
                "uploaded_count": len(processed_files),
                "total_sample_count": sample_count
            }
        )
        await db.logs.insert_one(log_entry.dict(exclude={"id"}))
        
        # Активируем пользователя, если достаточно образцов
        activation_message = ""
        if sample_count >= 5 and not user.get("active", False):
            await db.users.update_one(
                {"_id": ObjectId(user_id)},
                {"$set": {"active": True}}
            )
            
            # Обновление модели распознавания
            await call_ml_service("update_model", {
                "user_id": user_id,
                "embeddings": updated_user.get("voice_embeddings", [])
            })
            
            activation_message = "Пользователь был автоматически активирован."
            
            # Логирование активации
            log_entry = LogEntry(
                event_type="user_auto_activated",
                user_id=user_id,
                user_name=user.get("name"),
                success=True
            )
            await db.logs.insert_one(log_entry.dict(exclude={"id"}))
        
        return {
            "success": True,
            "processed_files": len(processed_files),
            "total_samples": sample_count,
            "activation_status": updated_user.get("active", False),
            "message": activation_message
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in batch upload: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Эндпоинты для экспорта/импорта системы
@app.get("/api/system/export")
async def export_system(db=Depends(get_db)):
    """
    Экспортирует всю систему в ZIP-архив
    """
    try:
        # Генерируем имя архива с датой
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        export_filename = f"voice_auth_export_{timestamp}.zip"
        export_path = os.path.join("/shared", export_filename)
        
        # Создаем временную директорию для подготовки данных
        temp_dir = tempfile.mkdtemp()
        
        try:
            # 1. Экспорт пользователей
            users = await db.users.find().to_list(1000)
            with open(os.path.join(temp_dir, "users.json"), "w") as f:
                json.dump(users, f, default=str)
            
            # 2. Экспорт образцов голоса
            voice_samples = await db.voice_samples.find().to_list(5000)
            with open(os.path.join(temp_dir, "voice_samples.json"), "w") as f:
                json.dump(voice_samples, f, default=str)
            
            # 3. Экспорт логов (опционально)
            logs = await db.logs.find().sort("timestamp", -1).limit(1000).to_list(1000)
            with open(os.path.join(temp_dir, "logs.json"), "w") as f:
                json.dump(logs, f, default=str)
            
            # 4. Создаем структуру для аудиофайлов
            audio_export_dir = os.path.join(temp_dir, "audio")
            os.makedirs(audio_export_dir, exist_ok=True)
            
            # Копируем аудиофайлы пользователей
            for user in users:
                user_id = str(user["_id"])
                user_dir = os.path.join(AUDIO_PATH, user_id)
                if os.path.exists(user_dir):
                    user_export_dir = os.path.join(audio_export_dir, user_id)
                    os.makedirs(user_export_dir, exist_ok=True)
                    
                    # Копируем все WAV файлы пользователя
                    for filename in os.listdir(user_dir):
                        if filename.lower().endswith('.wav'):
                            src_path = os.path.join(user_dir, filename)
                            dst_path = os.path.join(user_export_dir, filename)
                            shutil.copy2(src_path, dst_path)
            
            # 5. Создаем ZIP-архив
            shutil.make_archive(export_path[:-4], 'zip', temp_dir)
            
            # 6. Проверка, что файл успешно создан
            if not os.path.exists(export_path):
                raise Exception("Failed to create export file")
            
            return {
                "success": True,
                "export_file": export_filename,
                "message": "Система успешно экспортирована",
                "download_url": f"/download/{export_filename}"
            }
        
        finally:
            # Очистка временных файлов
            shutil.rmtree(temp_dir, ignore_errors=True)
    
    except Exception as e:
        logger.error(f"Error exporting system: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/download/{filename}")
async def download_file(filename: str):
    """
    Endpoint для скачивания экспортированных файлов
    """
    file_path = os.path.join("/shared", filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")
    
    return FileResponse(file_path, filename=filename)

# api/app.py - добавьте новые эндпоинты
@app.get("/api/auth/recent-events")
async def get_recent_auth_events(db=Depends(get_db)):
    """Получение недавних событий аутентификации"""
    try:
        # Получение последних событий аутентификации из базы
        events = await db.logs.find({
            "event_type": {"$in": ["authorization_successful", "authorization_attempt", "spoofing_attempt"]}
        }).sort("timestamp", -1).limit(10).to_list(10)
        
        formatted_events = []
        for event in events:
            event_type = "success" if event["event_type"] == "authorization_successful" else \
                        "spoof" if event["event_type"] == "spoofing_attempt" else "failure"
            
            user_info = None
            if event.get("user_id"):
                # Получение данных о пользователе, если есть user_id
                from bson.objectid import ObjectId
                user = await db.users.find_one({"_id": ObjectId(event["user_id"])})
                if user:
                    user_info = {
                        "id": str(user["_id"]),
                        "name": user.get("name", "Неизвестно")
                    }
            
            details = ""
            if event_type == "spoof":
                details = f"Вероятность спуфинга: {event.get('details', {}).get('spoof_probability', 0) * 100:.1f}%"
            
            formatted_events.append({
                "id": str(event["_id"]),
                "timestamp": event["timestamp"].isoformat(),
                "type": event_type,
                "user": user_info,
                "similarity": event.get("details", {}).get("similarity"),
                "details": details
            })
        
        return {"events": formatted_events}
    except Exception as e:
        logger.error(f"Error getting auth events: {e}")
        return {"events": [], "error": str(e)}

@app.post("/api/auth/settings")
async def update_auth_settings(
    settings: dict,
    db=Depends(get_db)
):
    """Обновление настроек аутентификации"""
    try:
        # Сохранение настроек в базе данных
        await db.settings.update_one(
            {"type": "auth_settings"},
            {"$set": {
                "similarity_threshold": float(settings.get("similarity_threshold", 0.75)),
                "spoof_sensitivity": float(settings.get("spoof_sensitivity", 0.5)),
                "auto_threshold": settings.get("auto_threshold", True),
                "updated_at": datetime.datetime.now()
            }},
            upsert=True
        )
        
        # Обновление настроек в ML-сервисе
        await call_ml_service("POST", "/update_auth_settings", settings)

        return {"success": True, "message": "Настройки успешно обновлены"}
    except Exception as e:
        logger.error(f"Error updating auth settings: {e}")
        return {"success": False, "error": str(e)}

# api/app.py - добавьте новый эндпоинт для диагностики файловой системы
@app.get("/api/system/filesystem")
async def check_filesystem(db=Depends(get_db)):
    """Проверка структуры файловой системы для аудиофайлов"""
    try:
        result = {
            "audio_path_exists": os.path.exists(AUDIO_PATH),
            "audio_path": AUDIO_PATH,
            "users_directories": [],
            "total_audio_files": 0,
            "missing_files": 0
        }
        
        # Проверка корневой директории
        if not os.path.exists(AUDIO_PATH):
            logger.error(f"Audio path does not exist: {AUDIO_PATH}")
            return result
        
        # Получение списка пользователей из базы данных
        users = await db.users.find().to_list(1000)
        
        # Проверка директорий пользователей и файлов
        for user in users:
            user_id = str(user["_id"])
            user_dir = os.path.join(AUDIO_PATH, user_id)
            
            user_info = {
                "user_id": user_id,
                "name": user.get("name", "Unknown"),
                "directory_exists": os.path.exists(user_dir),
                "directory_path": user_dir,
                "sample_count_db": len(user.get("voice_samples", [])),
                "sample_count_fs": 0,
                "files": []
            }
            
            # Если директория существует, проверим файлы
            if os.path.exists(user_dir):
                audio_files = [f for f in os.listdir(user_dir) if f.endswith((".wav", ".WAV"))]
                user_info["sample_count_fs"] = len(audio_files)
                result["total_audio_files"] += len(audio_files)
                
                # Получим список образцов голоса из базы данных
                samples = await db.voice_samples.find({"user_id": user_id}).to_list(100)
                
                # Проверим наличие файлов из базы данных
                for sample in samples:
                    file_path = sample.get("file_path", "")
                    filename = os.path.basename(file_path) if file_path else ""
                    
                    file_info = {
                        "sample_id": str(sample["_id"]),
                        "filename": filename,
                        "db_path": file_path,
                        "exists": os.path.exists(file_path) if file_path else False
                    }
                    
                    # Если файл не существует, проверим альтернативные пути
                    if not file_info["exists"] and filename:
                        alt_path = os.path.join(user_dir, filename)
                        if os.path.exists(alt_path):
                            file_info["alt_path"] = alt_path
                            file_info["alt_exists"] = True
                        else:
                            result["missing_files"] += 1
                    
                    user_info["files"].append(file_info)
            else:
                # Если директория не существует, все файлы отсутствуют
                samples = await db.voice_samples.find({"user_id": user_id}).to_list(100)
                result["missing_files"] += len(samples)
            
            result["users_directories"].append(user_info)
        
        return result
    except Exception as e:
        logger.error(f"Error checking filesystem: {e}")
        return {"error": str(e)}

# api/app.py - добавьте новые эндпоинты
@app.post("/api/system/fix_file_paths")
async def fix_file_paths(db=Depends(get_db)):
    """Исправление путей к файлам в базе данных"""
    try:
        # Получение всех образцов голоса
        samples = await db.voice_samples.find().to_list(10000)
        fixed_count = 0
        
        for sample in samples:
            file_path = sample.get("file_path")
            if not file_path or not os.path.exists(file_path):
                # Попытка найти файл по альтернативным путям
                user_id = sample.get("user_id")
                filename = os.path.basename(file_path) if file_path else ""
                
                if filename:
                    # Проверяем возможные пути
                    alternative_paths = [
                        os.path.join(AUDIO_PATH, user_id, filename),
                        os.path.join(AUDIO_PATH, str(user_id), filename)
                    ]
                    
                    for alt_path in alternative_paths:
                        if os.path.exists(alt_path):
                            # Обновляем путь в базе данных
                            await db.voice_samples.update_one(
                                {"_id": sample["_id"]},
                                {"$set": {"file_path": alt_path}}
                            )
                            fixed_count += 1
                            logger.info(f"Fixed path for sample {sample['_id']}: {alt_path}")
                            break
        
        return {
            "success": True,
            "fixed_count": fixed_count,
            "total_samples": len(samples)
        }
    except Exception as e:
        logger.error(f"Error fixing file paths: {e}")
        return {"success": False, "message": str(e)}

@app.post("/api/system/create_missing_dirs")
async def create_missing_dirs(db=Depends(get_db)):
    """Создание отсутствующих директорий пользователей"""
    try:
        # Получение всех пользователей
        users = await db.users.find().to_list(1000)
        created_count = 0
        
        # Проверка корневой директории
        if not os.path.exists(AUDIO_PATH):
            os.makedirs(AUDIO_PATH, exist_ok=True)
            logger.info(f"Created root audio directory: {AUDIO_PATH}")
        
        # Создание директорий пользователей
        for user in users:
            user_id = str(user["_id"])
            user_dir = os.path.join(AUDIO_PATH, user_id)
            
            if not os.path.exists(user_dir):
                os.makedirs(user_dir, exist_ok=True)
                created_count += 1
                logger.info(f"Created directory for user {user_id}: {user_dir}")
        
        return {
            "success": True,
            "created_count": created_count,
            "total_users": len(users)
        }
    except Exception as e:
        logger.error(f"Error creating missing directories: {e}")
        return {"success": False, "message": str(e)}

@app.post("/api/system/cleanup_data")
async def cleanup_data(db=Depends(get_db)):
    """Очистка неконсистентных данных в базе данных"""
    try:
        # Получение всех образцов голоса
        samples = await db.voice_samples.find().to_list(10000)
        removed_count = 0
        
        for sample in samples:
            file_path = sample.get("file_path")
            sample_id = str(sample["_id"])
            user_id = sample.get("user_id")
            
            # Проверка наличия файла
            file_exists = False
            if file_path and os.path.exists(file_path):
                file_exists = True
            else:
                # Проверка альтернативных путей
                filename = os.path.basename(file_path) if file_path else ""
                if filename:
                    alt_path = os.path.join(AUDIO_PATH, user_id, filename)
                    if os.path.exists(alt_path):
                        file_exists = True
            
            # Если файл не существует, удаляем запись
            if not file_exists:
                # Удаление образца из базы данных
                await db.voice_samples.delete_one({"_id": sample["_id"]})
                
                # Удаление ссылки на образец из пользователя
                await db.users.update_one(
                    {"_id": ObjectId(user_id)},
                    {"$pull": {"voice_samples": sample_id}}
                )
                
                removed_count += 1
                logger.info(f"Removed sample {sample_id} for user {user_id} due to missing file")
        
        # Проверка и деактивация пользователей с недостаточным количеством образцов
        users = await db.users.find({"active": True}).to_list(1000)
        deactivated_count = 0
        
        for user in users:
            user_id = str(user["_id"])
            sample_count = len(user.get("voice_samples", []))
            
            if sample_count < 5:
                await db.users.update_one(
                    {"_id": user["_id"]},
                    {"$set": {"active": False}}
                )
                deactivated_count += 1
                logger.info(f"Deactivated user {user_id} due to insufficient samples ({sample_count})")
        
        return {
            "success": True,
            "removed_count": removed_count,
            "deactivated_count": deactivated_count,
            "total_samples": len(samples)
        }
    except Exception as e:
        logger.error(f"Error cleaning up data: {e}")
        return {"success": False, "message": str(e)}

@app.post("/api/system/regenerate_all_embeddings")
async def regenerate_all_embeddings(db=Depends(get_db)):
    """Пересоздание эмбеддингов для всех аудиофайлов"""
    try:
        # Получение всех пользователей
        users = await db.users.find().to_list(1000)
        updated_count = 0
        error_count = 0
        
        for user in users:
            user_id = str(user["_id"])
            
            # Получение образцов голоса пользователя
            samples = await db.voice_samples.find({"user_id": user_id}).to_list(100)
            
            # Пересоздание эмбеддингов
            new_embeddings = []
            
            for sample in samples:
                file_path = sample.get("file_path")
                
                # Проверка существования файла
                if not file_path or not os.path.exists(file_path):
                    # Попытка найти файл по альтернативному пути
                    filename = os.path.basename(file_path) if file_path else ""
                    if filename:
                        alt_path = os.path.join(AUDIO_PATH, user_id, filename)
                        if os.path.exists(alt_path):
                            file_path = alt_path
                
                if file_path and os.path.exists(file_path):
                    try:
                        # Извлечение эмбеддинга
                        embedding_result = await call_ml_service("POST", "/extract_embedding", {
                            "audio_path": file_path
                        })
                        
                        if "embedding" in embedding_result:
                            embedding = embedding_result["embedding"]
                            
                            # Обновление эмбеддинга в образце
                            await db.voice_samples.update_one(
                                {"_id": sample["_id"]},
                                {"$set": {"embedding": embedding}}
                            )
                            
                            # Добавление в список новых эмбеддингов
                            new_embeddings.append(embedding)
                            updated_count += 1
                        else:
                            error_count += 1
                    except Exception as e:
                        logger.error(f"Error regenerating embedding for sample {sample['_id']}: {e}")
                        error_count += 1
                else:
                    error_count += 1
            
            # Обновление эмбеддингов пользователя
            if new_embeddings:
                await db.users.update_one(
                    {"_id": user["_id"]},
                    {"$set": {"voice_embeddings": new_embeddings}}
                )
                
                # Проверка необходимости активации пользователя
                if len(new_embeddings) >= 5 and not user.get("active", False):
                    await db.users.update_one(
                        {"_id": user["_id"]},
                        {"$set": {"active": True}}
                    )
                    logger.info(f"Activated user {user_id} after regenerating embeddings")
                
                # Обновление модели
                try:
                    await call_ml_service("POST", "/update_model", {
                        "user_id": user_id,
                        "embeddings": new_embeddings
                    })
                except Exception as e:
                    logger.error(f"Error updating ML model for user {user_id}: {e}")
        
        return {
            "success": True,
            "updated_count": updated_count,
            "error_count": error_count
        }
    except Exception as e:
        logger.error(f"Error regenerating all embeddings: {e}")
        return {"success": False, "message": str(e)}

# Фикция для работы с тестовым интерфейсом
@app.post("/api/test/authenticate")
async def test_authenticate(
    audio_file: UploadFile = File(...),
    threshold: float = Form(0.7),
    expected_user: Optional[str] = Form(None)
):
    """
    Тестовый эндпоинт для проверки аутентификации по аудиофайлу с настраиваемым порогом
    """
    try:
        logger.info(f"Received test authentication request with threshold={threshold}")
        
        # Проверка имени файла
        if audio_file.filename == '':
            return JSONResponse(
                status_code=400,
                content={"success": False, "message": "No selected file"}
            )
            
        # Создание временного файла с безопасным именем
        filename = audio_file.filename
        temp_file_path = os.path.join(UPLOAD_FOLDER, f"test_{uuid.uuid4()}_{filename}")
        
        # Сохранение файла
        try:
            with open(temp_file_path, "wb") as buffer:
                content = await audio_file.read()
                buffer.write(content)
        except Exception as file_error:
            logger.error(f"Error saving uploaded file: {file_error}")
            return JSONResponse(
                status_code=500,
                content={"success": False, "message": f"Error saving file: {str(file_error)}"}
            )
        
        # Проверка файла
        if not os.path.exists(temp_file_path):
            return JSONResponse(
                status_code=500,
                content={"success": False, "message": "Failed to save audio file"}
            )
            
        # Преобразование в WAV если необходимо
        final_file_path = temp_file_path
        if not filename.lower().endswith('.wav'):
            try:
                # Используем ffmpeg для конвертации в WAV
                converted_path = os.path.join(UPLOAD_FOLDER, f"test_{uuid.uuid4()}.wav")
                subprocess.run([
                    'ffmpeg', '-y', '-i', temp_file_path,
                    '-ar', '16000', '-ac', '1', '-f', 'wav', converted_path
                ], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                
                if os.path.exists(converted_path):
                    final_file_path = converted_path
                    logger.info(f"Converted audio file to WAV: {final_file_path}")
            except Exception as e:
                logger.warning(f"Failed to convert audio format: {e}")
                # Продолжаем с оригинальным файлом
        
        # Отправка запроса на спуфинг-детекцию в ML сервис
        try:
            # Подготовка файла для отправки
            with open(final_file_path, 'rb') as f:
                files = {'audio_data': (os.path.basename(final_file_path), f)}
                
                # Отправка запроса
                spoof_response = requests.post(
                    f"{ML_SERVICE_URL}/detect_spoofing", 
                    files=files
                )
                
                # Проверка ответа
                if spoof_response.status_code != 200:
                    logger.error(f"ML service returned status code: {spoof_response.status_code}")
                    return JSONResponse(
                        status_code=500,
                        content={"success": False, "message": f"ML service error: {spoof_response.text}"}
                    )
                    
                # Получение результата
                spoof_result = spoof_response.json()
                
                # Проверка на спуфинг
                is_spoof = spoof_result.get("is_spoof", False)
                spoof_prob = spoof_result.get("spoof_probability", 0.1)
                
                # Используем более строгий порог для предотвращения ложных срабатываний
                spoof_threshold = 0.4
                
                if is_spoof or spoof_prob > spoof_threshold:
                    logger.warning(f"Spoofing detected in test file: {spoof_prob:.4f}")
                    
                    # Сохраняем результат теста
                    test_result = {
                        "success": True,
                        "authorized": False,
                        "spoofing_detected": True,
                        "spoof_probability": float(spoof_prob),
                        "match_score": 0,
                        "user_id": None,
                        "threshold": float(threshold),
                        "test_file": os.path.basename(filename),
                        "expected_user": expected_user
                    }
                    
                    # Записываем результат в лог
                    await log_test_result(test_result)
                    
                    # Удаляем временные файлы
                    cleanup_temp_files([temp_file_path, final_file_path])
                    
                    return test_result
        except Exception as e:
            logger.error(f"Error during spoofing detection: {e}")
            # Продолжаем без спуфинг-детекции
        
        # Извлечение эмбеддинга и сопоставление с пользователями
        try:
            # Подготовка файла для отправки
            with open(final_file_path, 'rb') as f:
                files = {'audio_data': (os.path.basename(final_file_path), f)}
                
                # Отправка запроса на извлечение эмбеддинга
                embedding_response = requests.post(
                    f"{ML_SERVICE_URL}/extract_embedding", 
                    files=files
                )
                
                # Проверка ответа
                if embedding_response.status_code != 200:
                    logger.error(f"ML service returned status code: {embedding_response.status_code}")
                    return JSONResponse(
                        status_code=500,
                        content={"success": False, "message": f"ML service error: {embedding_response.text}"}
                    )
                    
                # Получение эмбеддинга
                embedding_result = embedding_response.json()
                embedding = embedding_result.get("embedding", [])
                
                # Отправка запроса на сопоставление с пользователями
                match_data = {
                    "embedding": embedding,
                    "threshold": float(threshold)
                }
                
                match_response = requests.post(
                    f"{ML_SERVICE_URL}/match_user_detailed",
                    json=match_data
                )
                
                # Проверка ответа
                if match_response.status_code != 200:
                    logger.error(f"ML service returned status code: {match_response.status_code}")
                    return JSONResponse(
                        status_code=500,
                        content={"success": False, "message": f"ML service error: {match_response.text}"}
                    )
                    
                # Получение результата
                match_result = match_response.json()
                
                # Формирование результата теста
                test_result = {
                    "success": True,
                    "authorized": match_result.get("match_found", False),
                    "user_id": match_result.get("user_id"),
                    "match_score": int(match_result.get("similarity", 0) * 100),
                    "spoofing_detected": False,
                    "similarity": float(match_result.get("similarity", 0)),
                    "threshold": float(match_result.get("threshold", threshold)),
                    "match_found": match_result.get("match_found", False),
                    "test_file": os.path.basename(filename),
                    "expected_user": expected_user
                }
                
                # Добавляем информацию о кандидатах на совпадение
                if "match_candidates" in match_result:
                    test_result["match_candidates"] = match_result["match_candidates"]
                    
                # Если есть детальная информация о сходстве
                if "detailed_similarity" in match_result:
                    test_result["detailed_similarity"] = match_result["detailed_similarity"]
                
                # Записываем результат в лог
                await log_test_result(test_result)
                
                # Удаляем временные файлы
                cleanup_temp_files([temp_file_path, final_file_path])
                
                return test_result
                
        except Exception as e:
            logger.error(f"Error during embedding extraction and matching: {e}")
            return JSONResponse(
                status_code=500,
                content={"success": False, "message": f"Processing error: {str(e)}"}
            )
    except Exception as e:
        logger.error(f"Error in test authentication: {e}")
        return JSONResponse(
            status_code=500,
            content={"success": False, "message": str(e)}
        )

def cleanup_temp_files(file_paths):
    """Удаляет временные файлы"""
    for path in file_paths:
        try:
            if os.path.exists(path) and path != file_paths[0]:  # Не удаляем исходный файл
                os.remove(path)
                logger.info(f"Removed temp file: {path}")
        except Exception as e:
            logger.warning(f"Failed to remove temp file {path}: {e}")
            
async def log_test_result(result):
    """Записывает результат теста в лог"""
    try:
        # Имя файла лога с текущей датой
        log_file = os.path.join(LOG_FOLDER, f"voice_auth_tests_{datetime.now().strftime('%Y-%m-%d')}.jsonl")
        
        # Добавляем метку времени к результату
        result["timestamp"] = datetime.now().isoformat()
        
        # Создаем директорию для логов, если её нет
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        
        # Записываем в файл в формате JSON Lines
        with open(log_file, "a") as f:
            f.write(json.dumps(result) + "\n")
            
        logger.info(f"Logged test result to {log_file}")
    except Exception as e:
        logger.error(f"Error logging test result: {e}")

@app.get("/api/users/list")
async def list_users():
    """
    Возвращает список пользователей для тестового интерфейса
    """
    try:
        # Проверяем подключение к базе данных
        if not db:
            return JSONResponse(
                status_code=500,
                content={"success": False, "message": "Database not connected"}
            )
            
        # Пытаемся получить пользователей из базы данных
        users_collection = db["users"]
        users = []
        
        try:
            # Получаем пользователей из MongoDB
            user_docs = await users_collection.find({}).to_list(length=100)
            
            for user_doc in user_docs:
                # Преобразуем ObjectId в строку
                user_id = str(user_doc.get("_id", ""))
                
                users.append({
                    "id": user_id,
                    "name": user_doc.get("name", user_id),
                    "embedding_count": user_doc.get("embedding_count", 0)
                })
                
            logger.info(f"Found {len(users)} users in database")
        except Exception as db_error:
            logger.error(f"Error getting users from MongoDB: {db_error}")
            # Продолжаем с получением списка из ML-сервиса
                
        # Если пользователей нет в базе или произошла ошибка, запрашиваем из ML-сервиса
        if not users:
            # Запрашиваем информацию из ML-сервиса
            try:
                response = requests.get(f"{ML_SERVICE_URL}/health")
                
                if response.status_code != 200:
                    logger.error(f"ML service returned status code: {response.status_code}")
                    return {
                        "success": False,
                        "message": f"ML service error: {response.text}"
                    }
                    
                # Получение информации о пользователях
                result = response.json()
                
                # Извлечение количества пользователей
                user_count = result.get("user_count", 0)
                
                # Если у нас есть пользователи, запрашиваем список ID
                if user_count > 0:
                    try:
                        user_ids_response = requests.get(f"{ML_SERVICE_URL}/list_user_ids")
                        if user_ids_response.status_code == 200:
                            user_ids = user_ids_response.json().get("user_ids", [])
                            
                            for user_id in user_ids:
                                users.append({
                                    "id": user_id,
                                    "name": user_id,
                                    "embedding_count": 0
                                })
                        else:
                            # Если эндпоинта нет, создаем заглушки на основе количества
                            for i in range(user_count):
                                users.append({
                                    "id": f"user_{i+1}",
                                    "name": f"Пользователь {i+1}",
                                    "embedding_count": 0
                                })
                    except Exception as e:
                        logger.warning(f"Error fetching user IDs: {e}")
                        # Создаем заглушки на основе количества
                        for i in range(user_count):
                            users.append({
                                "id": f"user_{i+1}",
                                "name": f"Пользователь {i+1}",
                                "embedding_count": 0
                            })
            except Exception as ml_error:
                logger.error(f"Error getting user count from ML service: {ml_error}")
        
        return {
            "success": True,
            "users": users,
            "total": len(users)
        }
        
    except Exception as e:
        logger.error(f"Error listing users: {e}")
        return JSONResponse(
            status_code=500,
            content={"success": False, "message": str(e)}
        )


# Для поддержки WebSocket-соединений
@app.websocket("/api/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket для передачи событий аутентификации в реальном времени"""
    await manager.connect(websocket)
    try:
        # Отправка последних событий для инициализации
        recent_events = await db.logs.find({
            "event_type": {"$in": ["authorization_successful", "authorization_attempt", "spoofing_attempt"]}
        }).sort("timestamp", -1).limit(10).to_list(10)

        for event in recent_events:
            event_data = format_auth_event(event)
            await websocket.send_json(event_data)

        # Ожидание сообщений для поддержания соединения
        while True:
            data = await websocket.receive_text()
            if data == "ping":
                await websocket.send_text("pong")
    except WebSocketDisconnect:
        manager.disconnect(websocket)


# Исправление: Функция должна быть объявлена как async
async def format_auth_event(event):
    """Форматирование события аутентификации для отправки через WebSocket"""
    from bson.objectid import ObjectId

    # Определение типа события
    event_type = "success" if event["event_type"] == "authorization_successful" else \
        "spoof" if event["event_type"] == "spoofing_attempt" else "failure"

    # Базовая информация
    formatted_event = {
        "id": str(event["_id"]),
        "timestamp": event["timestamp"].isoformat() if isinstance(event["timestamp"], datetime) else event[
            "timestamp"],
        "type": event_type,
        "user_id": event.get("user_id"),
        "similarity": event.get("details", {}).get("similarity")
    }

    # Добавление информации о пользователе
    if event.get("user_id"):
        user_info = None
        user = await db.users.find_one({"_id": ObjectId(event["user_id"])})
        if user:
            user_info = {
                "id": str(user["_id"]),
                "name": user.get("name", "Неизвестный пользователь"),
                "department": user.get("department", ""),
                "position": user.get("position", "")
            }
            formatted_event["user"] = user_info

    return formatted_event

# Эндпоинты для управления режимом КПП
@app.post("/api/kpp/start")
async def start_kpp_mode():
    """Запуск режима КПП"""
    try:
        # Уведомляем аудио-процессор
        response = await call_audio_processor("kpp_mode", {
            "enabled": True,
            "sensitivity": 0.03,  # Можно сделать настраиваемым
            "min_speech_duration": 1.5,
            "auto_reset_delay": 5,
            "callback_url": f"{os.environ.get('API_BASE_URL', 'http://api:5000')}/api/kpp/callback"
        })

        # Сохраняем информацию о состоянии в БД
        await db.system_status.update_one(
            {"_id": "kpp_status"},
            {"$set": {
                "active": True,
                "started_at": datetime.now(),
                "settings": {
                    "sensitivity": 0.03,
                    "min_speech_duration": 1.5,
                    "auto_reset_delay": 5
                }
            }},
            upsert=True
        )

        return {"status": "started", "message": "Режим КПП активирован"}
    except Exception as e:
        logger.error(f"Error starting KPP mode: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/kpp/stop")
async def stop_kpp_mode():
    """Остановка режима КПП"""
    try:
        # Отключение в аудио-процессоре
        response = await call_audio_processor("kpp_mode", {
            "enabled": False
        })

        # Обновление статуса в БД
        await db.system_status.update_one(
            {"_id": "kpp_status"},
            {"$set": {
                "active": False,
                "stopped_at": datetime.datetime.now()
            }}
        )

        return {"status": "stopped", "message": "Режим КПП деактивирован"}
    except Exception as e:
        logger.error(f"Error stopping KPP mode: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/kpp/status")
async def get_kpp_status(db=Depends(get_db)):
    """Проверка статуса режима КПП"""
    try:
        status = await db.system_status.find_one({"_id": "kpp_status"})

        # Если записи нет, режим не активен
        if not status:
            return {
                "active": False,
                "message": "Режим КПП никогда не был активирован"
            }

        # Форматирование для JSON
        status["_id"] = str(status["_id"])

        # Преобразуем даты в строки ISO
        for key in ["started_at", "stopped_at"]:
            if key in status and isinstance(status[key], datetime.datetime):
                status[key] = status[key].isoformat()

        return status
    except Exception as e:
        logger.error(f"Error getting KPP status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/kpp/callback")
async def kpp_callback(file_path: str):
    """Обработка аудио из режима КПП"""
    try:
        logger.info(f"Received KPP callback: file_path={file_path}")

        # Проверка существования файла
        if not os.path.exists(file_path):
            logger.error(f"Audio file not found: {file_path}")
            return {"success": False, "error": "Audio file not found"}

        # Извлечение эмбеддинга и проверка спуфинга через ML-модель
        try:
            # Сначала проверяем на спуфинг
            anti_spoof_result = await call_ml_service("POST", "/detect_spoofing", {
                "audio_path": file_path
            })

            if anti_spoof_result.get("is_spoof", False):
                # Обнаружена попытка подделки
                logger.warning(
                    f"Spoofing detected in file {file_path}: {anti_spoof_result.get('spoof_probability', 0)}")

                # Логируем событие
                event_data = {
                    "event_type": "spoofing_attempt",
                    "timestamp": datetime.datetime.now(),
                    "success": False,
                    "details": {
                        "audio_path": file_path,
                        "spoof_probability": anti_spoof_result.get("spoof_probability", 0),
                        "source": "kpp_callback"
                    }
                }

                # Сохраняем в БД
                result = await db.logs.insert_one(event_data)

                # Отправляем через WebSocket или другие механизмы оповещения
                # ...

                return {
                    "success": True,
                    "authorized": False,
                    "spoofing_detected": True,
                    "message": "Обнаружена попытка подделки голоса"
                }

            # Сравнение с базой голосов
            embedding_result = await call_ml_service("POST", "/extract_embedding", {
                "audio_path": file_path
            })

            embedding = embedding_result.get("embedding")

            if not embedding:
                logger.error("Failed to extract embedding from audio")
                return {"success": False, "error": "Failed to extract embedding"}

            # Сравнение с эмбеддингами всех пользователей
            match_result = await call_ml_service("POST", "/match_user_detailed", {
                "embedding": embedding
            })

            user_id = match_result.get("user_id")
            similarity = match_result.get("similarity", 0)
            match_found = match_result.get("match_found", False)

            # Готовим данные о событии аутентификации
            event_data = {
                "event_type": "authorization_successful" if match_found else "authorization_attempt",
                "timestamp": datetime.datetime.now(),
                "user_id": user_id,
                "success": match_found,
                "details": {
                    "similarity": similarity,
                    "match_score": int(similarity * 100),
                    "audio_path": file_path,
                    "source": "kpp_callback"
                }
            }

            # Сохраняем в БД
            result = await db.logs.insert_one(event_data)
            event_id = str(result.inserted_id)

            # Получаем информацию о пользователе если успешная аутентификация
            user_info = None
            if match_found and user_id:
                from bson.objectid import ObjectId
                user = await db.users.find_one({"_id": ObjectId(user_id)})

                if user:
                    user_info = {
                        "id": str(user["_id"]),
                        "name": user.get("name", "Неизвестный пользователь"),
                        "department": user.get("department", ""),
                        "position": user.get("position", ""),
                        "access_level": user.get("access_level", "standard")
                    }

                    # Проверка наличия фото
                    photo_path = os.path.join(AUDIO_PATH, user_id, "photo.jpg")
                    if os.path.exists(photo_path):
                        user_info["photo_url"] = f"/api/users/{user_id}/photo"

            auth_result = {
                "success": True,
                "authorized": match_found,
                "user": user_info,
                "similarity": similarity,
                "match_score": int(similarity * 100),
                "event_id": event_id,
                "spoofing_detected": False
            }

            logger.info(f"Authentication result: {auth_result}")

            # Отправка результата через WebSocket всем подключенным клиентам (если реализовано)
            # ...

            return auth_result

        except Exception as e:
            logger.error(f"Error processing audio for authentication: {e}")
            return {"success": False, "error": str(e)}
    except Exception as e:
        logger.error(f"Error in KPP callback: {e}")
        return {"success": False, "error": str(e)}


@app.post("/api/users/{user_id}/delete")
async def delete_user(
        user_id: str,
        db=Depends(get_db)
):
    """
    Удаление пользователя с полной очисткой связанных данных
    """
    from bson.objectid import ObjectId

    try:
        # Проверка существования пользователя
        user = await db.users.find_one({"_id": ObjectId(user_id)})
        if not user:
            raise HTTPException(status_code=404, detail="Пользователь не найден")

        # Логирование события удаления
        log_entry = LogEntry(
            event_type="user_deleted",
            user_id=user_id,
            user_name=user.get("name"),
            success=True,
            details={"method": "admin_delete"}
        )
        await db.logs.insert_one(log_entry.dict(exclude={"id"}))

        # Удаление голосовых образцов
        samples_result = await db.voice_samples.delete_many({"user_id": user_id})
        logger.info(f"Deleted {samples_result.deleted_count} voice samples for user {user_id}")

        # Удаление файлов пользователя
        user_dir = os.path.join(AUDIO_PATH, user_id)
        try:
            if os.path.exists(user_dir):
                shutil.rmtree(user_dir)
                logger.info(f"Deleted user directory: {user_dir}")
        except Exception as e:
            logger.error(f"Error deleting user directory {user_dir}: {e}")

        # Удаление пользователя из базы данных
        delete_result = await db.users.delete_one({"_id": ObjectId(user_id)})

        # Обновление ML-модели
        try:
            await call_ml_service("remove_user", {
                "user_id": user_id
            })
        except Exception as ml_error:
            logger.warning(f"Error updating ML model after user deletion: {ml_error}")

        return {
            "success": True,
            "message": "Пользователь успешно удален",
            "deleted_count": delete_result.deleted_count,
            "voice_samples_deleted": samples_result.deleted_count
        }
    except HTTPException:
        # Переопределяем HTTPException, чтобы они проходили без повторной обработки
        raise
    except Exception as e:
        logger.error(f"Error deleting user {user_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/system/reinitialize")
async def reinitialize_system():
    """Полная переинициализация системы"""
    try:
        # Перезагрузка эмбеддингов
        emb_success = force_reload_embeddings()

        # Сброс порогов
        global adaptive_threshold, SPOOFING_THRESHOLD
        adaptive_threshold = 0.5  # Установите низкий порог для тестирования
        SPOOFING_THRESHOLD = 0.7  # Высокий порог для спуфинга

        return {
            "success": True,
            "embeddings_reloaded": emb_success,
            "user_count": len(user_embeddings),
            "adaptive_threshold": adaptive_threshold,
            "spoofing_threshold": SPOOFING_THRESHOLD
        }
    except Exception as e:
        logger.error(f"Error reinitializing system: {e}")
        return {
            "success": False,
            "error": str(e)
        }

# Запуск сервера для локальной разработки
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000)