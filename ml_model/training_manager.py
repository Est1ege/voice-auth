# ml_model/training_manager.py
import os
import threading
import uuid
import json
import logging
import time
from datetime import datetime
from typing import Dict, List, Any, Optional
import torch

# Импорт классов тренеров
from voice_model_trainer import VoiceModelTrainer
from anti_spoof_trainer import AntiSpoofTrainer

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("training_manager")

class TrainingManager:
    """
    Менеджер для управления тренировками моделей
    """
    def __init__(self, model_path: str, audio_path: str):
        self.model_path = model_path
        self.audio_path = audio_path
        self.active_tasks = {}  # Активные задачи тренировки
        self.task_history = {}  # История задач
        
        # Создаем директории для тренировки моделей
        self.temp_model_path = os.path.join(model_path, "temp")
        os.makedirs(self.temp_model_path, exist_ok=True)
        
        # Директории для голосовых образцов
        self.real_voice_path = os.path.join(audio_path)
        
        # Директория для спуфинг образцов
        self.spoof_voice_path = os.path.join(audio_path, "_spoof_samples")
        os.makedirs(self.spoof_voice_path, exist_ok=True)
        
        # Использование GPU, если доступно
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Training manager initialized (device: {self.device})")
        
        # Загрузка истории задач, если существует
        self._load_task_history()
    
    def _load_task_history(self):
        """
        Загружает историю задач из файла
        """
        history_file = os.path.join(self.model_path, "training_history.json")
        if os.path.exists(history_file):
            try:
                with open(history_file, "r") as f:
                    self.task_history = json.load(f)
                logger.info(f"Loaded training history with {len(self.task_history)} tasks")
            except Exception as e:
                logger.error(f"Error loading task history: {e}")
                self.task_history = {}
    
    def _save_task_history(self):
        """
        Сохраняет историю задач в файл
        """
        history_file = os.path.join(self.model_path, "training_history.json")
        try:
            with open(history_file, "w") as f:
                json.dump(self.task_history, f, indent=4)
            logger.info(f"Saved training history with {len(self.task_history)} tasks")
        except Exception as e:
            logger.error(f"Error saving task history: {e}")

    def _get_all_users(self) -> List[str]:
        """
        Получает список ID всех пользователей из базы данных
        (Примечание: этот метод нужно реализовать в соответствии с вашей архитектурой)
        """
        try:
            # Этот код должен быть адаптирован к вашей базе данных и структуре данных
            # Например, можно импортировать и использовать функцию из API-сервиса

            # В качестве примера, просто сканируем директории с аудиофайлами
            user_ids = []
            if os.path.exists(self.real_voice_path):
                for item in os.listdir(self.real_voice_path):
                    item_path = os.path.join(self.real_voice_path, item)
                    if os.path.isdir(item_path) and item != "_spoof_samples":
                        user_ids.append(item)

            logger.info(f"Found {len(user_ids)} users for training")
            return user_ids
        except Exception as e:
            logger.error(f"Error getting user list: {e}")
            return []

    def _train_voice_model_thread(self, task_id: str, params: Dict[str, Any]):
        """
        Функция для запуска тренировки модели распознавания голоса в отдельном потоке
        """
        try:
            # Создание объекта тренера
            trainer = VoiceModelTrainer(
                model_path=self.model_path,
                audio_path=self.real_voice_path,
                output_path=os.path.join(self.temp_model_path, task_id),
                device=self.device,
                batch_size=params.get("batch_size", 32),
                learning_rate=params.get("learning_rate", 0.001),
                num_epochs=params.get("num_epochs", 50)
            )

            # Запуск тренировки и получение списка пользователей
            user_ids = self._get_all_users()
            if not user_ids:
                self.active_tasks[task_id]["status"] = "failed"
                self.active_tasks[task_id]["message"] = "No users found for training"
                self.active_tasks[task_id]["end_time"] = datetime.now().isoformat()
                # Сохранение в историю
                self.task_history[task_id] = self.active_tasks[task_id].copy()
                self._save_task_history()
                return

            # Запуск тренировки
            training_result = trainer.train(user_ids)

            # Обработка результата тренировки
            if isinstance(training_result, dict):
                # Новый формат возвращаемого значения (словарь)
                status = training_result.get("status", "error")
                message = training_result.get("message", "Unknown error")
                progress = training_result.get("progress", 0.0)

                self.active_tasks[task_id]["status"] = status
                self.active_tasks[task_id]["progress"] = progress
                self.active_tasks[task_id]["message"] = message
            elif isinstance(training_result, bool):
                # Старый формат возвращаемого значения (логическое значение)
                if training_result:
                    self.active_tasks[task_id]["status"] = "completed"
                    self.active_tasks[task_id]["progress"] = 1.0
                    self.active_tasks[task_id]["message"] = "Training completed successfully"
                else:
                    self.active_tasks[task_id]["status"] = "failed"
                    self.active_tasks[task_id]["progress"] = 0.0
                    self.active_tasks[task_id]["message"] = "Training failed"
            else:
                # Неизвестный формат возвращаемого значения
                self.active_tasks[task_id]["status"] = "error"
                self.active_tasks[task_id]["progress"] = 0.0
                self.active_tasks[task_id]["message"] = "Unknown training result type"

            # Устанавливаем время завершения
            self.active_tasks[task_id]["end_time"] = datetime.now().isoformat()

            # Сохранение в историю
            self.task_history[task_id] = self.active_tasks[task_id].copy()
            self._save_task_history()

            # Логирование результата
            if self.active_tasks[task_id]["status"] == "completed":
                logger.info(f"Voice model training task {task_id} completed successfully")
            else:
                logger.error(f"Voice model training task {task_id} failed: {self.active_tasks[task_id]['message']}")

        except Exception as e:
            logger.error(f"Error in voice model training thread for task {task_id}: {e}")
            self.active_tasks[task_id]["status"] = "error"
            self.active_tasks[task_id]["progress"] = 0.0
            self.active_tasks[task_id]["message"] = f"Error: {str(e)}"
            self.active_tasks[task_id]["end_time"] = datetime.now().isoformat()

            # Сохранение в историю
            self.task_history[task_id] = self.active_tasks[task_id].copy()
            self._save_task_history()

    def _train_anti_spoof_model_thread(self, task_id: str, params: Dict[str, Any]):
        """
        Функция для запуска тренировки модели защиты от спуфинга в отдельном потоке
        """
        try:
            # Создание объекта тренера
            trainer = AntiSpoofTrainer(
                model_path=self.model_path,
                real_audio_path=self.real_voice_path,
                spoof_audio_path=self.spoof_voice_path,
                output_path=os.path.join(self.temp_model_path, task_id),
                device=self.device,
                batch_size=params.get("batch_size", 32),
                learning_rate=params.get("learning_rate", 0.001),
                num_epochs=params.get("num_epochs", 50)
            )
            
            # Запуск тренировки
            result = trainer.train(task_id)
            
            # Обновление статуса задачи
            self.active_tasks[task_id]["status"] = result["status"]
            self.active_tasks[task_id]["progress"] = result["progress"]
            self.active_tasks[task_id]["message"] = result["message"]
            self.active_tasks[task_id]["end_time"] = datetime.now().isoformat()
            
            # Сохранение в историю
            self.task_history[task_id] = self.active_tasks[task_id].copy()
            self._save_task_history()
            
            # Удаление из активных задач
            if result["status"] == "completed":
                logger.info(f"Anti-spoof model training task {task_id} completed successfully")
            else:
                logger.error(f"Anti-spoof model training task {task_id} failed: {result['message']}")
            
        except Exception as e:
            logger.error(f"Error in anti-spoof model training thread for task {task_id}: {e}")
            self.active_tasks[task_id]["status"] = "error"
            self.active_tasks[task_id]["message"] = f"Error: {str(e)}"
            self.active_tasks[task_id]["end_time"] = datetime.now().isoformat()
            
            # Сохранение в историю
            self.task_history[task_id] = self.active_tasks[task_id].copy()
            self._save_task_history()
    
    def start_voice_model_training(self, params: Dict[str, Any] = None) -> str:
        """
        Запускает тренировку модели распознавания голоса
        
        Args:
            params: Параметры тренировки (batch_size, learning_rate, num_epochs)
        
        Returns:
            str: ID задачи тренировки
        """
        if params is None:
            params = {}
        
        # Генерация чистого UUID без префикса для совместимости с API
        task_id = str(uuid.uuid4())
        
        # Создание записи о задаче
        self.active_tasks[task_id] = {
            "task_id": task_id,
            "type": "voice_model",  # Тип задается отдельным полем
            "status": "starting",
            "progress": 0.0,
            "message": "Starting voice model training",
            "start_time": datetime.now().isoformat(),
            "end_time": None,
            "params": params
        }
        
        # Запуск тренировки в отдельном потоке
        thread = threading.Thread(
            target=self._train_voice_model_thread,
            args=(task_id, params)
        )
        thread.daemon = True
        thread.start()
        
        logger.info(f"Started voice model training task {task_id}")
        
        return task_id
    
    def start_anti_spoof_model_training(self, params: Dict[str, Any] = None) -> str:
        """
        Запускает тренировку модели защиты от спуфинга
        
        Args:
            params: Параметры тренировки (batch_size, learning_rate, num_epochs)
        
        Returns:
            str: ID задачи тренировки
        """
        if params is None:
            params = {}
        
        # Генерация ID задачи
        task_id = str(uuid.uuid4())
        
        # Создание записи о задаче
        self.active_tasks[task_id] = {
            "task_id": task_id,
            "type": "anti_spoof",
            "status": "starting",
            "progress": 0.0,
            "message": "Starting anti-spoof model training",
            "start_time": datetime.now().isoformat(),
            "end_time": None,
            "params": params
        }
        
        # Запуск тренировки в отдельном потоке
        thread = threading.Thread(
            target=self._train_anti_spoof_model_thread,
            args=(task_id, params)
        )
        thread.daemon = True
        thread.start()
        
        logger.info(f"Started anti-spoof model training task {task_id}")
        
        return task_id

    def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """
        Возвращает статус задачи тренировки с улучшенной обработкой ошибок

        Args:
            task_id: ID задачи

        Returns:
            Dict: Информация о задаче или заглушка с ошибкой, если задача не найдена
        """
        try:
            # Проверка валидности task_id
            if not task_id or not isinstance(task_id, str):
                self.logger.warning(f"Invalid task_id: {task_id}")
                return {
                    "task_id": task_id if task_id else "unknown",
                    "status": "error",
                    "progress": 0.0,
                    "message": "Invalid task ID",
                    "type": None
                }

            # Проверка в активных задачах
            if task_id in self.active_tasks:
                return self.active_tasks[task_id]

            # Проверка в истории задач
            if task_id in self.task_history:
                return self.task_history[task_id]

            # Если задача не найдена, возвращаем информацию об этом
            self.logger.warning(f"Task not found: {task_id}")
            return {
                "task_id": task_id,
                "status": "unknown",
                "progress": 0.0,
                "message": f"Task with ID {task_id} not found",
                "type": None
            }
        except Exception as e:
            # Обработка любых других ошибок
            self.logger.error(f"Error getting training status: {e}")
            return {
                "task_id": task_id if task_id else "unknown",
                "status": "error",
                "progress": 0.0,
                "message": f"Error retrieving task status: {str(e)}",
                "type": None
            }
    
    def get_all_tasks(self) -> Dict[str, List[Dict[str, Any]]]:
        """
        Возвращает информацию о всех задачах (активных и из истории)
        
        Returns:
            Dict: Словарь с активными задачами и историей
        """
        return {
            "active": list(self.active_tasks.values()),
            "history": list(self.task_history.values())
        }
    
    def clean_completed_tasks(self, max_age_days: int = 7) -> int:
        """
        Очищает завершенные задачи из истории старше указанного возраста
        
        Args:
            max_age_days: Максимальный возраст задач в днях
        
        Returns:
            int: Количество удаленных задач
        """
        current_time = datetime.now()
        tasks_to_remove = []
        
        for task_id, task in self.task_history.items():
            if task["end_time"]:
                task_end_time = datetime.fromisoformat(task["end_time"])
                age_days = (current_time - task_end_time).days
                
                if age_days > max_age_days:
                    tasks_to_remove.append(task_id)
        
        # Удаление задач
        for task_id in tasks_to_remove:
            del self.task_history[task_id]
        
        # Сохранение обновленной истории
        if tasks_to_remove:
            self._save_task_history()
        
        return len(tasks_to_remove)