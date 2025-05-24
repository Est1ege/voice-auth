# ml_model/app.py
from fastapi import FastAPI, HTTPException, BackgroundTasks, APIRouter
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from anti_spoof import AntiSpoofingDetector
from pydantic import BaseModel
from typing import Any, Dict, List, Optional, Set, Tuple, Union
from scipy.spatial.distance import cosine
import numpy as np
import json
import torch
import logging
import time
import os

from pathlib import Path
import shutil
from datetime import datetime

# Импорт наших моделей
from voice_embedding import VoiceEmbeddingModel
from anti_spoof import AntiSpoofingDetector
from training_manager import TrainingManager  # Добавляем импорт менеджера тренировок


# Функция для безопасной сериализации NumPy значений
def safe_serialize(obj):
    """
    Безопасно преобразует объекты NumPy в обычные типы Python для сериализации JSON
    """
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, dict):
        return {safe_serialize(k): safe_serialize(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [safe_serialize(item) for item in obj]
    else:
        return obj


# Определение класса для безопасной сериализации
class CustomJSONResponse(JSONResponse):
    def render(self, content: Any) -> bytes:
        """
        Переопределяем метод render для безопасной сериализации NumPy типов
        """
        content = safe_serialize(content)
        return super().render(content)


# Создание экземпляра FastAPI
app = FastAPI(title="ML-сервис голосовой аутентификации", version="1.0.0", default_response_class=CustomJSONResponse)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
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
        logging.FileHandler("/app/logs/ml_service.log"),
    ]
)
logger = logging.getLogger("ml_service")

# Интеграция с audio_processor
try:
    import audio_processor_integration as api
except ImportError:
    logger.warning("Audio processor integration not available")

# Конфигурация
MODEL_PATH = os.environ.get("MODEL_PATH", "/shared/models")
EMBEDDINGS_PATH = os.environ.get("EMBEDDINGS_PATH", "/shared/embeddings")
AUDIO_PATH = os.environ.get("AUDIO_PATH", "/shared/audio")

# Создание директорий, если не существуют
os.makedirs(MODEL_PATH, exist_ok=True)
os.makedirs(EMBEDDINGS_PATH, exist_ok=True)
os.makedirs(AUDIO_PATH, exist_ok=True)
os.makedirs("/app/logs", exist_ok=True)

# Загрузка моделей
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

# Инициализация моделей
voice_model = None
anti_spoof_model = None
user_embeddings = {}

# Инициализация менеджера тренировок
training_manager = None


# Модели данных

class AudioRequest(BaseModel):
    audio_path: str


class SpoofingResponse(BaseModel):
    is_spoof: bool
    spoof_probability: float


class EmbeddingResponse(BaseModel):
    embedding: List[float]


class MatchRequest(BaseModel):
    embedding: List[float]


class MatchResponse(BaseModel):
    user_id: Optional[str] = None
    similarity: float
    match_found: bool


class UserEmbeddingRequest(BaseModel):
    user_id: str
    embeddings: List[List[float]]


class StatusResponse(BaseModel):
    status: str
    message: str
    models_loaded: bool
    device: str
    user_count: int


class TrainingRequest(BaseModel):
    model_type: str  # "voice_model" или "anti_spoof"
    batch_size: Optional[int] = 32
    learning_rate: Optional[float] = 0.001
    num_epochs: Optional[int] = 50


class TrainingResponse(BaseModel):
    task_id: str
    status: str
    message: str


class TrainingStatusRequest(BaseModel):
    task_id: str


class TrainingStatusResponse(BaseModel):
    task_id: str
    type: str
    status: str
    progress: float
    message: str
    start_time: Optional[str] = None
    end_time: Optional[str] = None


reinitialization_status = {
    "in_progress": False,
    "last_status": None,
    "start_time": None,
    "end_time": None,
    "progress": 0.0
}


# Найдите функцию load_user_embeddings
def load_user_embeddings():
    global user_embeddings
    embeddings_file = os.path.join(EMBEDDINGS_PATH, "user_embeddings.json")

    if os.path.exists(embeddings_file):
        try:
            with open(embeddings_file, "r") as f:
                user_embeddings = json.load(f)

            # Преобразуем строковые представления списков обратно в numpy массивы
            for user_id, embeddings in user_embeddings.items():
                user_embeddings[user_id] = [np.array(emb) for emb in embeddings]

            logger.info(f"Loaded embeddings for {len(user_embeddings)} users")

            # Добавляем детальную диагностику
            for user_id, embeddings in user_embeddings.items():
                quality = "good" if any(
                    not np.all(np.isnan(emb)) and not np.all(emb == 0) for emb in embeddings) else "bad"
                logger.info(f"User {user_id}: {len(embeddings)} embeddings, quality: {quality}")
        except Exception as e:
            logger.error(f"Error loading user embeddings: {e}")
            user_embeddings = {}
    else:
        logger.info("No user embeddings file found. Starting with empty database.")
        user_embeddings = {}


# Сохранение эмбеддингов пользователей
def save_user_embeddings():
    global user_embeddings
    embeddings_file = os.path.join(EMBEDDINGS_PATH, "user_embeddings.json")

    try:
        # Создаем копию с преобразованием numpy массивов в списки
        serializable_embeddings = {}
        for user_id, embeddings in user_embeddings.items():
            serializable_embeddings[user_id] = [emb.tolist() for emb in embeddings]

        with open(embeddings_file, "w") as f:
            json.dump(serializable_embeddings, f)

        logger.info(f"Saved embeddings for {len(user_embeddings)} users")
    except Exception as e:
        logger.error(f"Error saving user embeddings: {e}")


def force_reload_embeddings():
    """Принудительная перезагрузка эмбеддингов с глубокой проверкой"""
    global user_embeddings

    embeddings_file = os.path.join(EMBEDDINGS_PATH, "user_embeddings.json")

    if not os.path.exists(embeddings_file):
        logger.warning(f"Embeddings file not found at {embeddings_file}")
        user_embeddings = {}
        return False

    try:
        # Создание бэкапа перед изменениями
        backup_file = os.path.join(EMBEDDINGS_PATH, f"user_embeddings_backup_{int(time.time())}.json")
        try:
            shutil.copy2(embeddings_file, backup_file)
            logger.info(f"Created backup at {backup_file}")
        except Exception as e:
            logger.warning(f"Failed to create backup: {e}")

        # Загрузка и глубокая проверка эмбеддингов
        with open(embeddings_file, "r") as f:
            raw_embeddings = json.load(f)

        # Новый словарь для проверенных эмбеддингов
        verified_embeddings = {}

        # Проверка и коррекция эмбеддингов для каждого пользователя
        for user_id, emb_list in raw_embeddings.items():
            valid_embeddings = []

            for emb in emb_list:
                try:
                    # Преобразование в numpy массив
                    np_emb = np.array(emb)

                    # Проверка размерности
                    if len(np_emb.shape) != 1:
                        logger.warning(f"Unexpected embedding shape for user {user_id}: {np_emb.shape}, reshaping...")
                        np_emb = np_emb.flatten()

                    # Проверка на NaN и Inf
                    if np.any(np.isnan(np_emb)) or np.any(np.isinf(np_emb)):
                        logger.warning(f"Invalid values in embedding for user {user_id}, skipping")
                        continue

                    # Проверка на нулевой вектор
                    if np.all(np_emb == 0):
                        logger.warning(f"Zero embedding for user {user_id}, skipping")
                        continue

                    # Нормализация
                    norm = np.linalg.norm(np_emb)
                    if norm < 1e-10:
                        logger.warning(f"Near-zero norm for user {user_id}, skipping")
                        continue

                    np_emb = np_emb / norm
                    valid_embeddings.append(np_emb)

                except Exception as e:
                    logger.warning(f"Error processing embedding for user {user_id}: {e}")
                    continue

            # Добавляем пользователя только если есть валидные эмбеддинги
            if valid_embeddings:
                verified_embeddings[user_id] = valid_embeddings
                logger.info(f"User {user_id}: {len(valid_embeddings)} valid embeddings (from {len(emb_list)})")
            else:
                logger.warning(f"User {user_id}: No valid embeddings found!")

        # Замена старых эмбеддингов на проверенные
        user_embeddings = verified_embeddings
        logger.info(f"Reloaded embeddings for {len(user_embeddings)} users (from {len(raw_embeddings)})")

        return True

    except Exception as e:
        logger.error(f"Critical error reloading embeddings: {e}")
        return False


def check_audio_quality(audio_path):
    """Безопасная проверка качества аудиофайла"""
    try:
        # Импортируем необходимые библиотеки
        import librosa
        import numpy as np

        # Загружаем аудио через librosa
        try:
            waveform, sr = librosa.load(audio_path, sr=16000, mono=True)
        except Exception as e:
            logger.error(f"Error loading audio for quality check: {e}")
            return True, 0.5, 1.0, 10.0  # Предполагаем, что аудио приемлемого качества

        # Простая проверка энергии
        energy = np.mean(np.abs(waveform))

        # Обнаружение речи через обрезку тишины
        y_trimmed, _ = librosa.effects.trim(waveform, top_db=20)
        speech_duration = len(y_trimmed) / sr

        # Проверка минимальной длительности
        if speech_duration < 0.5 or energy < 0.005:
            return False, energy, speech_duration, 0.0

        # Оценка SNR (простой вариант)
        noise_sample = waveform[:int(sr * 0.1)]  # Первые 100 мс
        noise_power = np.mean(np.abs(noise_sample) ** 2)
        signal_power = np.mean(np.abs(waveform) ** 2)

        if noise_power > 0:
            snr = 10 * np.log10(signal_power / noise_power)
        else:
            snr = 30.0  # Высокий SNR, если шум не обнаружен

        # Проверка SNR
        if snr < -10:
            return False, energy, speech_duration, snr

        return True, energy, speech_duration, snr

    except Exception as e:
        logger.error(f"Error in audio quality check: {e}")
        return True, 0.02, 1.0, 10.0  # По умолчанию считаем, что речь есть


# Инициализация моделей при запуске
@app.on_event("startup")
async def startup_event():
    global voice_model, anti_spoof_model, training_manager
    # В функцию startup_event():
    if os.path.exists(os.path.join(MODEL_PATH, "voice_embedding_model", "ecapa_tdnn.pt")):
        model_stats = os.stat(os.path.join(MODEL_PATH, "voice_embedding_model", "ecapa_tdnn.pt"))
        logger.info(
            f"Voice model size: {model_stats.st_size} bytes, modified: {datetime.fromtimestamp(model_stats.st_mtime)}")
    try:
        # Инициализация модели извлечения эмбеддингов голоса
        voice_embedding_model_path = os.path.join(MODEL_PATH, "voice_embedding_model")
        voice_model = VoiceEmbeddingModel(model_path=voice_embedding_model_path, device=device)
        logger.info("Voice embedding model initialized")
        force_reload_embeddings()
        # Инициализация модели обнаружения спуфинга
        anti_spoofing_model_path = os.path.join(MODEL_PATH, "anti_spoofing_model")
        anti_spoof_model = AntiSpoofingDetector(model_path=anti_spoofing_model_path, device=device)
        logger.info("Anti-spoofing model initialized")

        # Загрузка эмбеддингов пользователей
        load_user_embeddings()

        # Инициализация менеджера тренировок
        training_manager = TrainingManager(MODEL_PATH, AUDIO_PATH)
        logger.info("Training manager initialized")

    except Exception as e:
        logger.error(f"Error initializing models: {e}")
        # Не останавливаем сервер, но логируем ошибку
        # В реальном приложении здесь могла бы быть более сложная обработка ошибок


@app.on_event("shutdown")
async def shutdown_event():
    # Сохраняем эмбеддинги пользователей перед завершением работы
    save_user_embeddings()
    logger.info("ML service shutting down")


# Эндпоинты API
@app.get("/health")
@app.post("/health")
async def health_check():
    global voice_model, anti_spoof_model

    models_loaded = voice_model is not None and anti_spoof_model is not None

    return StatusResponse(
        status="ok",
        message="ML service is running",
        models_loaded=models_loaded,
        device=str(device),
        user_count=len(user_embeddings)
    )


@app.post("/detect_spoofing")
async def detect_spoofing(request: AudioRequest):
    """
    Обнаружение спуфинга в аудиофайле.
    """
    try:
        logger.info(f"Detecting spoofing in audio file: {request.audio_path}")

        # Проверка существования файла с подробным логированием
        file_exists = os.path.exists(request.audio_path)
        logger.info(f"File exists check: {file_exists}")

        if not file_exists:
            # Проверка директории, где должен быть файл
            directory = os.path.dirname(request.audio_path)
            if os.path.exists(directory):
                logger.info(f"Directory exists: {directory}")
                logger.info(f"Directory contents: {os.listdir(directory)}")
            else:
                logger.info(f"Directory does not exist: {directory}")

            # Проверка общей директории
            shared_temp = "/shared/temp"
            if os.path.exists(shared_temp):
                logger.info(f"Shared temp exists: {shared_temp}")
                logger.info(f"Shared temp contents: {os.listdir(shared_temp)}")

            return {
                "is_spoof": False,
                "spoof_probability": 0.0,
                "confidence": 0.5,
                "error": "Audio file not found"
            }

        # Используем нашу модель для обнаружения спуфинга
        result = anti_spoof_model.detect(request.audio_path)

        # Логируем результат и преобразуем все numpy типы в обычные Python типы
        logger.info(f"Spoofing detection result: {result}")

        # Безопасно сериализуем результат
        safe_result = safe_serialize(result)
        return safe_result

    except Exception as e:
        logger.error(f"Error in spoofing detection: {str(e)}")
        return {
            "is_spoof": False,
            "spoof_probability": 0.0,
            "confidence": 0.5,
            "error": str(e)
        }


# В app.py - более продвинутый подход с проверкой качества аудио
@app.post("/extract_embedding", response_model=EmbeddingResponse)
async def extract_embedding(request: AudioRequest):
    global voice_model

    logger.info(f"Extracting embedding from file: {request.audio_path}")
    logger.info(f"File exists check: {os.path.exists(request.audio_path)}")

    if voice_model is None:
        raise HTTPException(status_code=500, detail="Voice embedding model not loaded")

    try:
        # Проверка существования файла
        if not os.path.exists(request.audio_path):
            logger.error(f"Audio file not found: {request.audio_path}")
            raise HTTPException(status_code=404, detail=f"Audio file not found: {request.audio_path}")

        # Вызываем extract_embedding напрямую без проверки качества
        # (проверка будет выполнена внутри метода)
        embedding = voice_model.extract_embedding(request.audio_path)

        if embedding is None:
            raise HTTPException(status_code=400,
                                detail="Could not extract embedding from audio file. Check audio quality.")

        logger.info(f"Extracted embedding with shape {embedding.shape}")

        return EmbeddingResponse(
            embedding=embedding.tolist()
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error extracting embedding: {e}")
        raise HTTPException(status_code=500, detail=f"Error extracting embedding: {str(e)}")


@app.get("/list_user_ids")
async def list_user_ids():
    """
    Возвращает список ID пользователей в системе
    """
    global user_embeddings

    try:
        # Формирование списка ID пользователей
        user_ids = list(user_embeddings.keys())

        return {
            "success": True,
            "user_ids": user_ids,
            "count": len(user_ids)
        }
    except Exception as e:
        logger.error(f"Error listing user IDs: {e}")
        return {
            "success": False,
            "message": str(e)
        }


@app.post("/match_user", response_model=MatchResponse)
async def match_user(request: MatchRequest):
    global user_embeddings

    try:
        # Преобразование входящего эмбеддинга в numpy массив
        input_embedding = np.array(request.embedding)

        if len(user_embeddings) == 0:
            logger.warning("No user embeddings available for matching")
            return MatchResponse(
                match_found=False,
                similarity=0.0,
                user_id=None
            )

        # Поиск наилучшего соответствия
        best_match_user_id = None
        best_match_similarity = 0.0

        # Пониженный порог для тестирования
        match_threshold = 0.5

        for user_id, embeddings in user_embeddings.items():
            logger.info(f"Comparing with user {user_id}: {len(embeddings)} embeddings")

            for i, user_embedding in enumerate(embeddings):
                try:
                    # Проверка размерностей
                    if user_embedding.shape[0] != input_embedding.shape[0]:
                        logger.warning(
                            f"Dimension mismatch for user {user_id}, embedding {i}: {user_embedding.shape} vs {input_embedding.shape}")
                        continue

                    # Нормализация эмбеддингов перед сравнением
                    input_norm = np.linalg.norm(input_embedding)
                    user_norm = np.linalg.norm(user_embedding)

                    if input_norm < 1e-10 or user_norm < 1e-10:
                        logger.warning(f"Near-zero norm for embedding comparison: input={input_norm}, user={user_norm}")
                        continue

                    input_normalized = input_embedding / input_norm
                    user_normalized = user_embedding / user_norm

                    # Вычисление косинусного сходства
                    cosine_similarity = np.dot(input_normalized, user_normalized)

                    # Преобразование в диапазон [0, 1]
                    similarity = float((cosine_similarity + 1) / 2)

                    logger.info(f"User {user_id}, embedding {i}: similarity={similarity:.4f}")

                    if similarity > best_match_similarity:
                        best_match_similarity = similarity
                        best_match_user_id = user_id

                except Exception as e:
                    logger.error(f"Error comparing with user {user_id}, embedding {i}: {e}")
                    continue

        # Пороговое значение для определения совпадения
        match_found = best_match_similarity >= match_threshold

        logger.info(
            f"Best match: user_id={best_match_user_id}, similarity={best_match_similarity:.4f}, match_found={match_found}")

        # Применение безопасной сериализации к результату
        return MatchResponse(
            user_id=str(best_match_user_id) if match_found and best_match_user_id else None,
            similarity=float(best_match_similarity),
            match_found=bool(match_found)
        )

    except Exception as e:
        logger.error(f"Error in user matching: {e}")
        # Вывод подробной информации об ошибке для отладки
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/update_model")
async def update_model(request: UserEmbeddingRequest, background_tasks: BackgroundTasks):
    global user_embeddings

    try:
        # Преобразование входящих эмбеддингов в numpy массивы
        embeddings = [np.array(emb) for emb in request.embeddings]

        # Обновление эмбеддингов пользователя
        user_embeddings[request.user_id] = embeddings

        # Сохранение эмбеддингов в фоновом режиме
        background_tasks.add_task(save_user_embeddings)

        logger.info(f"Updated embeddings for user_id={request.user_id}, count={len(embeddings)}")

        return {"success": True, "message": "User embeddings updated"}
    except Exception as e:
        logger.error(f"Error updating user embeddings: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Новые эндпоинты для тренировки моделей
@app.post("/training/start", response_model=TrainingResponse)
async def start_training(request: TrainingRequest):
    global training_manager

    if training_manager is None:
        raise HTTPException(status_code=500, detail="Training manager not initialized")

    try:
        params = {
            "batch_size": request.batch_size,
            "learning_rate": request.learning_rate,
            "num_epochs": request.num_epochs
        }

        if request.model_type == "voice_model":
            task_id = training_manager.start_voice_model_training(params)
        elif request.model_type == "anti_spoof":
            task_id = training_manager.start_anti_spoof_model_training(params)
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported model type: {request.model_type}")

        # Получение начального статуса
        status = training_manager.get_task_status(task_id)

        return TrainingResponse(
            task_id=task_id,
            status=status["status"],
            message=status["message"]
        )
    except Exception as e:
        logger.error(f"Error starting training: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Добавьте этот эндпоинт
@app.get("/training/{task_id}/status")
async def get_task_status(task_id: str):
    global training_manager

    logger.info(f"Received request for training status: task_id={task_id}")

    if training_manager is None:
        logger.error("Training manager not initialized")
        return {
            "task_id": task_id,
            "status": "error",
            "progress": 0.0,
            "message": "Training manager not initialized",
            "type": None
        }

    try:
        status = training_manager.get_task_status(task_id)
        logger.info(f"Retrieved status for task {task_id}: {status['status']}")
        return status
    except Exception as e:
        logger.error(f"Error getting training status: {e}")
        return {
            "task_id": task_id,
            "status": "error",
            "progress": 0.0,
            "message": f"Error retrieving task status: {str(e)}",
            "type": None
        }


@app.post("/training/{task_id}/cancel")
async def cancel_training_task(task_id: str):
    """
    Отменяет активную задачу тренировки
    """
    global training_manager

    if training_manager is None:
        raise HTTPException(status_code=500, detail="Training manager not initialized")

    try:
        success = training_manager.cancel_task(task_id)

        if success:
            return {
                "success": True,
                "message": f"Training task {task_id} cancelled successfully"
            }
        else:
            return {
                "success": False,
                "message": f"Failed to cancel task {task_id}. Task may not exist or already completed."
            }

    except Exception as e:
        logger.error(f"Error cancelling training task {task_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/training/stats")
async def get_training_statistics():
    """
    Возвращает статистику по тренировкам
    """
    global training_manager

    if training_manager is None:
        return {
            "active_tasks": 0,
            "completed_tasks": 0,
            "failed_tasks": 0,
            "voice_model_tasks": 0,
            "anti_spoof_tasks": 0
        }

    try:
        all_tasks = training_manager.get_all_tasks()

        stats = {
            "total_tasks": len(all_tasks),
            "active_tasks": len(
                [t for t in all_tasks if t.get("status") in ["starting", "preparing_data", "training"]]),
            "completed_tasks": len([t for t in all_tasks if t.get("status") == "completed"]),
            "failed_tasks": len([t for t in all_tasks if t.get("status") in ["error", "cancelled"]]),
            "voice_model_tasks": len([t for t in all_tasks if t.get("type") == "voice_model"]),
            "anti_spoof_tasks": len([t for t in all_tasks if t.get("type") == "anti_spoof"])
        }

        return stats

    except Exception as e:
        logger.error(f"Error getting training statistics: {e}")
        return {
            "error": str(e)
        }



@app.get("/training/list")
async def list_all_training_tasks():
    """
    Возвращает список всех задач тренировки (активных и завершенных)
    """
    global training_manager

    if training_manager is None:
        logger.error("Training manager not initialized")
        return []

    try:
        all_tasks = training_manager.get_all_tasks()

        # Добавляем дополнительную информацию для каждой задачи
        for task in all_tasks:
            # Форматируем времена для отображения
            if task.get("start_time"):
                try:
                    start_dt = datetime.fromisoformat(task["start_time"])
                    task["start_time_formatted"] = start_dt.strftime("%Y-%m-%d %H:%M:%S")
                except:
                    task["start_time_formatted"] = task["start_time"]

            if task.get("end_time"):
                try:
                    end_dt = datetime.fromisoformat(task["end_time"])
                    task["end_time_formatted"] = end_dt.strftime("%Y-%m-%d %H:%M:%S")

                    # Вычисляем длительность
                    if task.get("start_time"):
                        start_dt = datetime.fromisoformat(task["start_time"])
                        duration = end_dt - start_dt
                        task["duration"] = str(duration).split('.')[0]  # Убираем микросекунды
                except:
                    task["end_time_formatted"] = task["end_time"]

            # Добавляем понятное описание статуса
            status_descriptions = {
                "starting": "Запуск",
                "preparing_data": "Подготовка данных",
                "training": "Обучение",
                "completed": "Завершено",
                "error": "Ошибка",
                "cancelled": "Отменено"
            }
            task["status_description"] = status_descriptions.get(task.get("status"), task.get("status", "Неизвестно"))

            # Добавляем описание типа модели
            type_descriptions = {
                "voice_model": "Модель распознавания голоса",
                "anti_spoof": "Модель защиты от спуфинга"
            }
            task["type_description"] = type_descriptions.get(task.get("type"), task.get("type", "Неизвестно"))

        logger.info(f"Returned {len(all_tasks)} training tasks")
        return all_tasks

    except Exception as e:
        logger.error(f"Error listing training tasks: {e}")
        return []


@app.delete("/training/cleanup")
async def cleanup_old_training_tasks(max_age_days: int = 7):
    """
    Очищает старые завершенные задачи тренировки
    """
    global training_manager

    if training_manager is None:
        return {
            "success": False,
            "message": "Training manager not initialized",
            "removed_count": 0
        }

    try:
        removed_count = training_manager.clean_completed_tasks(max_age_days)

        return {
            "success": True,
            "removed_count": removed_count,
            "message": f"Removed {removed_count} old training tasks"
        }

    except Exception as e:
        logger.error(f"Error cleaning up training tasks: {e}")
        return {
            "success": False,
            "message": str(e),
            "removed_count": 0
        }

@app.post("/train_anti_spoofing")
async def train_anti_spoofing(background_tasks: BackgroundTasks):
    global anti_spoof_model

    if anti_spoof_model is None:
        raise HTTPException(status_code=500, detail="Anti-spoofing model not loaded")

    try:
        # Этот эндпоинт будет запускать обучение модели анти-спуфинга
        # в фоновом режиме, если необходимо
        # В этой демоверсии просто возвращаем успех

        return {"success": True, "message": "Anti-spoofing model training started"}
    except Exception as e:
        logger.error(f"Error starting anti-spoofing training: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Заменить текущую функцию в ml_model/app.py на эту
@app.post("/match_user_detailed")
async def match_user_detailed(request: dict):
    """Подробное сравнение эмбеддинга с базой пользователей с улучшенной надежностью"""
    global user_embeddings, voice_model

    try:
        embedding = request.get("embedding")

        if not embedding:
            logger.error("No embedding provided in request")
            return {
                "success": False,
                "message": "Embedding is required"
            }

        # Проверка структуры эмбеддинга
        try:
            np_embedding = np.array(embedding)

            # Проверка на нулевой эмбеддинг или NaN
            if np.all(np_embedding == 0) or np.any(np.isnan(np_embedding)):
                logger.error("Received invalid embedding (zero or NaN), can't perform matching")
                return {
                    "success": False,
                    "message": "Invalid embedding: contains zeros or NaNs",
                    "user_id": None,
                    "similarity": 0.0,
                    "match_found": False
                }

            if len(np_embedding.shape) != 1:
                logger.warning(f"Unexpected embedding shape: {np_embedding.shape}, reshaping...")
                np_embedding = np_embedding.flatten()

            # Нормализация входного эмбеддинга
            norm = np.linalg.norm(np_embedding)
            if norm > 0:
                np_embedding = np_embedding / norm

            # Проверка размерности
            expected_dim = 192  # Ожидаемая размерность эмбеддинга голоса
            if np_embedding.shape[0] != expected_dim:
                logger.warning(f"Unexpected embedding dimension: {np_embedding.shape[0]}, expected {expected_dim}")
                return {
                    "success": False,
                    "message": f"Invalid embedding dimension: expected {expected_dim}, got {np_embedding.shape[0]}"
                }
        except Exception as e:
            logger.error(f"Error processing embedding: {e}")
            return {
                "success": False,
                "message": f"Invalid embedding format: {str(e)}"
            }

        # Проверяем, что у нас есть эмбеддинги пользователей
        if len(user_embeddings) == 0:
            logger.warning("No user embeddings available for matching")
            return {
                "success": True,
                "user_id": None,
                "similarity": 0.0,
                "detailed_similarity": {
                    "weighted_score": 0.0,
                    "cosine_similarity": 0.0
                },
                "match_found": False,
                "message": "No users registered in the system"
            }

        # Поиск наилучшего соответствия с более гибким порогом
        best_match_user_id = None
        best_match_similarity = 0.0
        best_match_comparison = None
        all_results = []

        # Динамически определяем порог в зависимости от количества пользователей
        # Используем более высокий порог для избежания ложных срабатываний
        adaptive_threshold = max(0.5, min(0.75, 0.45 + 0.02 * len(user_embeddings)))
        logger.info(f"Using adaptive threshold: {adaptive_threshold:.2f} for {len(user_embeddings)} users")

        # Создаем словарь для хранения данных о совпадениях по пользователям
        user_match_data = {}

        for user_id, embeddings in user_embeddings.items():
            user_best_similarity = 0
            user_best_comparison = None
            user_all_scores = []

            # Проверяем, что у пользователя есть валидные эмбеддинги
            valid_embeddings = [emb for emb in embeddings if not np.all(emb == 0) and not np.any(np.isnan(emb))]

            if not valid_embeddings:
                logger.warning(f"User {user_id} has no valid embeddings, skipping")
                continue

            # Логируем количество эмбеддингов для пользователя
            logger.info(f"Comparing with user {user_id}: {len(valid_embeddings)} valid embeddings")

            # Улучшенная логика сравнения с использованием всех эмбеддингов пользователя
            for i, user_embedding in enumerate(valid_embeddings):
                try:
                    # Проверка размерности эмбеддинга пользователя
                    if user_embedding.shape[0] != np_embedding.shape[0]:
                        logger.warning(
                            f"Dimension mismatch: input {np_embedding.shape[0]}, user {user_embedding.shape[0]}"
                        )
                        continue

                    # Нормализация эмбеддинга пользователя
                    norm_user = np.linalg.norm(user_embedding)
                    if norm_user < 1e-10:
                        logger.warning(f"Near-zero norm for user embedding, skipping")
                        continue

                    user_embedding_norm = user_embedding / norm_user

                    # Косинусное сходство нормализованных векторов
                    raw_similarity = np.dot(np_embedding, user_embedding_norm)

                    # Улучшенное преобразование сходства с более агрессивными параметрами
                    def sigmoid_transform(x, center=0.75, sharpness=12):
                        """Улучшенное сигмоидальное преобразование для усиления различий"""
                        return 1.0 / (1.0 + np.exp(-sharpness * (x - center)))

                    adjusted_similarity = sigmoid_transform(raw_similarity)

                    # Проверка на случайное совпадение с более низким порогом
                    if abs(raw_similarity) < 0.25:
                        continue

                    # Нормализация в диапазон [0, 1] для обратной совместимости
                    cosine_similarity = (raw_similarity + 1) / 2

                    # Сохраняем результат в понятном формате
                    comparison = {
                        "raw_similarity": float(raw_similarity),
                        "adjusted_similarity": float(adjusted_similarity),
                        "cosine_similarity": float(cosine_similarity),
                        "weighted_score": float(adjusted_similarity)  # Основной показатель
                    }

                    # Сохраняем все сравнения для этого пользователя
                    user_all_scores.append({
                        "index": i,
                        "embedding": user_embedding,
                        "similarity": adjusted_similarity,
                        "comparison": comparison
                    })

                    # Логируем каждое сравнение для отладки
                    logger.info(f"User {user_id}, emb {i}: raw={raw_similarity:.4f}, adj={adjusted_similarity:.4f}")

                    # Сохраняем лучшее соответствие для этого пользователя
                    if adjusted_similarity > user_best_similarity:
                        user_best_similarity = adjusted_similarity
                        user_best_comparison = comparison

                except Exception as e:
                    logger.warning(f"Error comparing with user {user_id}, emb {i}: {e}")
                    continue

            # Применяем дополнительную проверку на согласованность эмбеддингов
            if len(user_all_scores) >= 3 and user_best_similarity > 0.6:
                # Сортируем по уменьшению сходства
                user_all_scores.sort(key=lambda x: x["similarity"], reverse=True)

                # Берем топ-3 лучших совпадения
                top_scores = user_all_scores[:3]

                # Проверяем разницу между ними - меньшая разница = более надежное совпадение
                if (top_scores[0]["similarity"] - top_scores[-1]["similarity"]) < 0.15:
                    # Эмбеддинги согласованы, увеличиваем оценку
                    consistency_bonus = 0.08  # 8% бонус
                    adjusted_similarity = min(1.0, user_best_similarity + consistency_bonus)
                    logger.info(f"Applied consistency bonus for user {user_id}: +{consistency_bonus:.2f}")
                    user_best_similarity = adjusted_similarity

            # Добавляем лучшее совпадение пользователя, если оно было найдено
            if user_best_comparison:
                # Сохраняем данные о совпадениях этого пользователя
                user_match_data[user_id] = {
                    "similarity": user_best_similarity,
                    "comparison": user_best_comparison,
                    "valid_embedding_count": len(valid_embeddings),
                    "match_scores": user_all_scores
                }

                # Добавляем лучшее сходство пользователя в общие результаты
                all_results.append({
                    "user_id": user_id,
                    "similarity": user_best_similarity,
                    "comparison": user_best_comparison
                })

                # Обновляем общее лучшее совпадение
                if user_best_similarity > best_match_similarity:
                    best_match_similarity = user_best_similarity
                    best_match_user_id = user_id
                    best_match_comparison = user_best_comparison

        # Сортируем результаты по убыванию сходства
        all_results.sort(key=lambda x: x["similarity"], reverse=True)

        # Формируем топ кандидатов
        top_candidates = []
        for result in all_results[:3]:  # Выводим топ-3 результата
            top_candidates.append({
                "user_id": result["user_id"],
                "similarity": float(result["similarity"])
            })
            logger.info(f"Match candidate: user_id={result['user_id']}, similarity={result['similarity']:.4f}")

        # Анализ конкурирующих кандидатов
        if len(all_results) >= 2:
            # Если разница между двумя лучшими кандидатами очень мала, проверяем дополнительные условия
            if (all_results[0]["similarity"] - all_results[1]["similarity"]) < 0.04:
                # Близкие кандидаты - проверяем количество согласованных эмбеддингов
                top_user = all_results[0]["user_id"]
                second_user = all_results[1]["user_id"]

                # Проверяем количество эмбеддингов
                if user_match_data[top_user]["valid_embedding_count"] < user_match_data[second_user][
                    "valid_embedding_count"]:
                    # Второй пользователь имеет больше эмбеддингов - предпочитаем его
                    if all_results[1]["similarity"] > adaptive_threshold - 0.05:  # Немного снижаем требования
                        logger.info(f"Preferring user {second_user} with more embeddings over {top_user}")
                        best_match_user_id = second_user
                        best_match_similarity = all_results[1]["similarity"]
                        best_match_comparison = all_results[1]["comparison"]

        # Используем адаптивный порог и добавляем логирование
        logger.info(
            f"Best match: user_id={best_match_user_id}, similarity={best_match_similarity:.4f}, threshold={adaptive_threshold:.4f}")

        if best_match_user_id and best_match_similarity >= adaptive_threshold:
            logger.info(f"Match found: user_id={best_match_user_id}, similarity={best_match_similarity:.4f}")
            detailed_result = {
                "success": True,
                "user_id": best_match_user_id,
                "similarity": float(best_match_similarity),
                "detailed_similarity": {
                    "cosine_similarity": float(best_match_comparison["cosine_similarity"]),
                    "weighted_score": float(best_match_comparison["weighted_score"])
                },
                "match_candidates": top_candidates,
                "threshold": float(adaptive_threshold),
                "match_found": True
            }
        else:
            logger.info(f"No match found meeting threshold={adaptive_threshold:.4f}")
            detailed_result = {
                "success": True,
                "user_id": None,
                "similarity": float(best_match_similarity) if best_match_comparison else 0.0,
                "detailed_similarity": {
                    "cosine_similarity": float(
                        best_match_comparison["cosine_similarity"]) if best_match_comparison else 0.0,
                    "weighted_score": float(best_match_comparison["weighted_score"]) if best_match_comparison else 0.0
                },
                "match_candidates": top_candidates,
                "threshold": float(adaptive_threshold),
                "match_found": False
            }

        return detailed_result

    except Exception as e:
        logger.error(f"Error in detailed user matching: {e}")
        return {
            "success": False,
            "message": str(e),
            "error_type": str(type(e).__name__)
        }


@app.post('/match_embedding')
async def match_embedding(request: MatchRequest):
    global user_embeddings

    if not user_embeddings:
        return MatchResponse(user_id=None, similarity=0.0, match_found=False)

    def normalize(vec):
        norm = np.linalg.norm(vec)
        return vec / norm if norm > 0 else vec

    input_emb = normalize(np.array(request.embedding))
    best_user = None
    best_score = -1.0

    for user_id, embeddings in user_embeddings.items():
        for stored_emb in embeddings:
            score = 1 - cosine(input_emb, normalize(np.array(stored_emb)))
            if score > best_score:
                best_score = score
                best_user = user_id

    threshold = 0.7  # Порог можно регулировать
    if best_score >= threshold:
        return MatchResponse(user_id=best_user, similarity=best_score, match_found=True)
    else:
        return MatchResponse(user_id=None, similarity=best_score, match_found=False)


@app.post("/reload_embeddings")
async def reload_embeddings():
    """Перезагрузка эмбеддингов пользователей из файла"""
    try:
        # Сначала сохраняем текущие эмбеддинги (на случай ошибки)
        backup_path = os.path.join(EMBEDDINGS_PATH, "user_embeddings_backup.json")
        try:
            # Создаем копию с преобразованием numpy массивов в списки
            serializable_embeddings = {}
            for user_id, embeddings in user_embeddings.items():
                serializable_embeddings[user_id] = [emb.tolist() for emb in embeddings]

            with open(backup_path, "w") as f:
                json.dump(serializable_embeddings, f)

            logger.info(f"Backed up embeddings for {len(user_embeddings)} users")
        except Exception as e:
            logger.warning(f"Error creating backup: {e}")

        # Загружаем эмбеддинги заново
        load_user_embeddings()

        return {
            "success": True,
            "user_count": len(user_embeddings),
            "message": f"Successfully reloaded embeddings for {len(user_embeddings)} users"
        }
    except Exception as e:
        logger.error(f"Error reloading embeddings: {e}")
        return {
            "success": False,
            "message": str(e)
        }


@app.get("/ping")
async def ping():
    """Простой эндпоинт для проверки доступности ML-сервиса"""
    return {"status": "ok", "message": "ML service is available"}


@app.get("/list_user_ids")
async def list_user_ids():
    """
    Возвращает список ID пользователей в системе
    """
    global user_embeddings

    try:
        # Формирование списка ID пользователей
        user_ids = list(user_embeddings.keys())

        return {
            "success": True,
            "user_ids": user_ids,
            "count": len(user_ids)
        }
    except Exception as e:
        logger.error(f"Error listing user IDs: {e}")
        return {
            "success": False,
            "message": str(e)
        }


# Добавьте в ml_model/app.py:

@app.post("/authenticate_by_path")
async def authenticate_by_path(request: dict):
    """Аутентификация пользователя по пути к аудиофайлу с улучшенной надежностью"""
    global voice_model, anti_spoof_model, user_embeddings

    if not voice_model or not anti_spoof_model:
        return {
            "success": False,
            "message": "Models not initialized"
        }

    try:
        audio_path = request.get("audio_path")

        if not audio_path or not os.path.exists(audio_path):
            logger.error(f"Audio file not found: {audio_path}")
            return {
                "success": False,
                "message": "Audio file not found"
            }

        # Проверка на спуфинг
        spoof_result = anti_spoof_model.detect(audio_path)
        is_spoof = spoof_result.get("is_spoofing_detected", False) or spoof_result.get("is_spoof", False)
        spoof_prob = spoof_result.get("spoof_probability", 0.1)

        # Используем более строгий порог для предотвращения ложных срабатываний
        spoof_threshold = 0.4  # Снизили с 0.6 для большей чувствительности

        if is_spoof or spoof_prob > spoof_threshold:
            logger.warning(f"Spoofing detected in file {audio_path}: {spoof_prob:.4f}")
            return {
                "success": True,
                "authorized": False,
                "spoofing_detected": True,
                "spoof_probability": float(spoof_prob),
                "match_score": 0,
                "user_id": None
            }

        # Извлечение эмбеддинга
        embedding = voice_model.extract_embedding(audio_path)

        if embedding is None:
            logger.error("Failed to extract embedding")
            return {
                "success": False,
                "message": "Failed to extract embedding"
            }

        # Создаем структуру для вызова функции match_user_detailed
        match_request = {"embedding": embedding.tolist()}

        # Получаем подробные результаты сравнения
        match_result = await match_user_detailed(match_request)

        # Проверяем результат
        if not match_result.get("success", False):
            return {
                "success": False,
                "message": match_result.get("message", "Error during matching process")
            }

        # Преобразуем результат в формат аутентификации
        auth_result = {
            "success": True,
            "authorized": match_result.get("match_found", False),
            "user_id": match_result.get("user_id"),
            "match_score": int(match_result.get("similarity", 0) * 100),
            "spoofing_detected": False,
            "similarity": float(match_result.get("similarity", 0)),
            "threshold": float(match_result.get("threshold", 0.7)),
            "match_found": match_result.get("match_found", False)
        }

        # Добавляем информацию о кандидатах на совпадение для отладки
        if "match_candidates" in match_result:
            auth_result["match_candidates"] = match_result["match_candidates"]

        # Если есть детальная информация о сходстве
        if "detailed_similarity" in match_result:
            auth_result["detailed_similarity"] = match_result["detailed_similarity"]

        return auth_result

    except Exception as e:
        logger.error(f"Error in authentication by path: {e}")
        return {
            "success": False,
            "message": str(e)
        }

@app.get("/system/reinitialize/status")
async def get_reinitialize_status():
    """Возвращает текущий статус переинициализации"""
    global reinitialization_status
    return reinitialization_status

# Модифицируйте функцию переинициализации
@app.post("/system/reinitialize")
def reinitialize_system():
    """Полная переинициализация системы"""
    global reinitialization_status

    # Если уже идет процесс, возвращаем текущий статус
    if reinitialization_status["in_progress"]:
        return {
            "success": False,
            "error": "Переинициализация уже выполняется",
            "status": reinitialization_status
        }

    try:
        # Устанавливаем статус "в процессе"
        reinitialization_status = {
            "in_progress": True,
            "last_status": "starting",
            "start_time": datetime.now().isoformat(),
            "end_time": None,
            "progress": 0.0,
            "steps": [
                {"name": "reload_embeddings", "status": "pending", "progress": 0},
                {"name": "reset_thresholds", "status": "pending", "progress": 0},
                {"name": "cleanup_models", "status": "pending", "progress": 0}
            ]
        }

        # Запускаем переинициализацию в фоновом потоке
        background_tasks.add_task(perform_reinitialize)

        return {
            "success": True,
            "message": "Переинициализация запущена",
            "status": reinitialization_status
        }
    except Exception as e:
        logger.error(f"Error starting system reinitialization: {e}")
        reinitialization_status = {
            "in_progress": False,
            "last_status": "error",
            "error": str(e),
            "end_time": datetime.now().isoformat(),
            "progress": 0.0
        }
        return {
            "success": False,
            "error": str(e),
            "status": reinitialization_status
        }
# Функция фонового выполнения
async def perform_reinitialize():
    global reinitialization_status, adaptive_threshold, SPOOFING_THRESHOLD

    try:
        # Шаг 1: Перезагрузка эмбеддингов
        reinitialization_status["last_status"] = "reloading_embeddings"
        reinitialization_status["progress"] = 0.2
        reinitialization_status["steps"][0]["status"] = "in_progress"

        emb_success = force_reload_embeddings()

        reinitialization_status["steps"][0]["status"] = "completed" if emb_success else "failed"
        reinitialization_status["steps"][0]["progress"] = 100

        # Шаг 2: Сброс порогов
        reinitialization_status["last_status"] = "resetting_thresholds"
        reinitialization_status["progress"] = 0.5
        reinitialization_status["steps"][1]["status"] = "in_progress"

        # Сброс порогов
        adaptive_threshold = 0.5
        SPOOFING_THRESHOLD = 0.7

        reinitialization_status["steps"][1]["status"] = "completed"
        reinitialization_status["steps"][1]["progress"] = 100

        # Шаг 3: Очистка моделей (если нужно)
        reinitialization_status["last_status"] = "cleaning_up"
        reinitialization_status["progress"] = 0.8
        reinitialization_status["steps"][2]["status"] = "in_progress"

        # Здесь можно добавить дополнительную очистку...

        reinitialization_status["steps"][2]["status"] = "completed"
        reinitialization_status["steps"][2]["progress"] = 100

        # Завершение
        reinitialization_status["in_progress"] = False
        reinitialization_status["last_status"] = "completed"
        reinitialization_status["end_time"] = datetime.now().isoformat()
        reinitialization_status["progress"] = 1.0

    except Exception as e:
        logger.error(f"Error during system reinitialization: {e}")
        reinitialization_status["in_progress"] = False
        reinitialization_status["last_status"] = "error"
        reinitialization_status["error"] = str(e)
        reinitialization_status["end_time"] = datetime.now().isoformat()
@app.post("/reset_embeddings")
async def reset_embeddings():
    """Сброс всех эмбеддингов и повторная загрузка из файла"""
    global user_embeddings

    try:
        # Создание резервной копии
        embeddings_file = os.path.join(EMBEDDINGS_PATH, "user_embeddings.json")
        backup_file = os.path.join(EMBEDDINGS_PATH, f"user_embeddings_backup_{int(time.time())}.json")

        if os.path.exists(embeddings_file):
            try:
                # Копируем файл в бэкап
                shutil.copy2(embeddings_file, backup_file)
                logger.info(f"Created backup at {backup_file}")
            except Exception as e:
                logger.warning(f"Error creating backup: {e}")

        # Очищаем текущие эмбеддинги
        old_count = len(user_embeddings)
        user_embeddings = {}

        # Перезагружаем из файла
        load_user_embeddings()

        # Возвращаем результат
        return {
            "success": True,
            "old_user_count": old_count,
            "new_user_count": len(user_embeddings),
            "message": f"Embeddings reset: {len(user_embeddings)} users loaded"
        }
    except Exception as e:
        logger.error(f"Error resetting embeddings: {e}")
        return {
            "success": False,
            "message": str(e)
        }


@app.post("/rebuild_user_embeddings")
async def rebuild_user_embeddings(user_id: str = None):
    """
    Пересоздает эмбеддинги для одного или всех пользователей из аудиофайлов
    """
    global voice_model, user_embeddings

    try:
        if not voice_model:
            return {
                "success": False,
                "message": "Voice model not initialized"
            }

        # Если указан конкретный пользователь, пересоздаем только его эмбеддинги
        if user_id:
            user_ids = [user_id]
            logger.info(f"Rebuilding embeddings for user: {user_id}")
        else:
            # Иначе пересоздаем эмбеддинги для всех пользователей
            user_ids = list(user_embeddings.keys())
            logger.info(f"Rebuilding embeddings for all {len(user_ids)} users")

        # Сохраняем текущие эмбеддинги как резервную копию
        backup_file = os.path.join(EMBEDDINGS_PATH, f"user_embeddings_backup_{int(time.time())}.json")
        try:
            # Создаем копию с преобразованием numpy массивов в списки
            serializable_embeddings = {}
            for uid, embeddings in user_embeddings.items():
                serializable_embeddings[uid] = [emb.tolist() for emb in embeddings]

            with open(backup_file, "w") as f:
                json.dump(serializable_embeddings, f)

            logger.info(f"Created backup of current embeddings at {backup_file}")
        except Exception as e:
            logger.warning(f"Error creating backup: {e}")

        # Счетчики для статистики
        total_users_processed = 0
        total_files_processed = 0
        users_updated = 0

        # Обрабатываем каждого пользователя
        for uid in user_ids:
            try:
                # Получаем аудиофайлы пользователя
                user_audio_files = []
                user_dir = os.path.join(AUDIO_PATH, uid)

                # Проверяем, существует ли директория пользователя
                if os.path.exists(user_dir) and os.path.isdir(user_dir):
                    # Собираем все WAV файлы
                    for filename in os.listdir(user_dir):
                        if filename.lower().endswith((".wav", ".mp3", ".flac")):
                            user_audio_files.append(os.path.join(user_dir, filename))

                # Если найдены аудиофайлы, обрабатываем их
                if user_audio_files:
                    logger.info(f"Found {len(user_audio_files)} audio files for user {uid}")

                    # Создаем список для новых эмбеддингов
                    new_embeddings = []

                    # Извлекаем эмбеддинги из каждого файла
                    for audio_file in user_audio_files:
                        try:
                            # Извлекаем эмбеддинг
                            embedding = voice_model.extract_embedding(audio_file)

                            # Если успешно извлечен, добавляем его
                            if embedding is not None and not np.any(np.isnan(embedding)) and not np.any(
                                    np.isinf(embedding)):
                                new_embeddings.append(embedding)
                                total_files_processed += 1
                                logger.info(f"Successfully extracted embedding from {audio_file}")
                            else:
                                logger.warning(f"Failed to extract valid embedding from {audio_file}")
                        except Exception as file_error:
                            logger.error(f"Error processing file {audio_file}: {file_error}")

                    # Если удалось извлечь хотя бы один эмбеддинг, обновляем пользователя
                    if new_embeddings:
                        user_embeddings[uid] = new_embeddings
                        users_updated += 1
                        logger.info(f"Updated user {uid} with {len(new_embeddings)} new embeddings")
                    else:
                        logger.warning(f"No valid embeddings extracted for user {uid}")
                else:
                    logger.warning(f"No audio files found for user {uid}")

                total_users_processed += 1

            except Exception as user_error:
                logger.error(f"Error processing user {uid}: {user_error}")

        # Сохраняем обновленные эмбеддинги
        save_user_embeddings()

        return {
            "success": True,
            "users_processed": total_users_processed,
            "users_updated": users_updated,
            "files_processed": total_files_processed,
            "message": f"Successfully rebuilt embeddings for {users_updated} users from {total_files_processed} files"
        }

    except Exception as e:
        logger.error(f"Error rebuilding user embeddings: {e}")
        return {
            "success": False,
            "message": str(e)
        }


@app.route('/model/status', methods=['GET'])
def model_status():
    """Возвращает статус ML моделей"""
    try:
        global model, anti_spoof_model, user_embeddings

        status = {
            "voice_model": {
                "loaded": model is not None,
                "type": type(model).__name__ if model else "Not loaded",
                "users_count": len(user_embeddings) if user_embeddings else 0
            },
            "anti_spoof_model": {
                "loaded": anti_spoof_model is not None,
                "type": type(anti_spoof_model).__name__ if anti_spoof_model else "Not loaded"
            },
            "embeddings": {
                "total_users": len(user_embeddings) if user_embeddings else 0,
                "total_embeddings": sum(len(embs) for embs in user_embeddings.values()) if user_embeddings else 0
            },
            "system": {
                "device": "GPU" if torch.cuda.is_available() else "CPU",
                "memory_usage": get_memory_usage(),
                "models_ready": (model is not None) and (anti_spoof_model is not None)
            }
        }

        return jsonify(status)
    except Exception as e:
        logger.error(f"Error getting model status: {e}")
        return jsonify({
            "error": str(e),
            "voice_model": {"loaded": False},
            "anti_spoof_model": {"loaded": False},
            "embeddings": {"total_users": 0, "total_embeddings": 0},
            "system": {"models_ready": False}
        }), 500


def get_memory_usage():
    """Получает информацию об использовании памяти"""
    try:
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.get_device_properties(0).total_memory
            gpu_allocated = torch.cuda.memory_allocated(0)
            gpu_reserved = torch.cuda.memory_reserved(0)

            return {
                "gpu_total": f"{gpu_memory // (1024 ** 3):.1f}GB",
                "gpu_allocated": f"{gpu_allocated // (1024 ** 3):.1f}GB",
                "gpu_reserved": f"{gpu_reserved // (1024 ** 3):.1f}GB",
                "gpu_percent": f"{(gpu_allocated / gpu_memory) * 100:.1f}%"
            }
        else:
            import psutil
            memory = psutil.virtual_memory()
            return {
                "ram_used": f"{memory.used // (1024 ** 3):.1f}GB",
                "ram_total": f"{memory.total // (1024 ** 3):.1f}GB",
                "ram_percent": f"{memory.percent:.1f}%"
            }
    except Exception as e:
        logger.error(f"Error getting memory usage: {e}")
        return {"error": str(e)}


# Запуск сервера для локальной разработки
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=5000)  # ml_model/app.py