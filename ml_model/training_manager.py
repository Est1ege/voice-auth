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
        """
        try:
            user_ids = []
            if os.path.exists(self.real_voice_path):
                for item in os.listdir(self.real_voice_path):
                    item_path = os.path.join(self.real_voice_path, item)
                    if os.path.isdir(item_path) and item != "_spoof_samples":
                        # Проверяем, есть ли в папке аудиофайлы
                        audio_files = [f for f in os.listdir(item_path) if f.endswith(('.wav', '.mp3', '.flac'))]
                        if audio_files:
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
            logger.info(f"Starting voice model training thread for task {task_id}")

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

            # Получение списка пользователей
            user_ids = self._get_all_users()
            if not user_ids:
                self.active_tasks[task_id]["status"] = "error"
                self.active_tasks[task_id]["message"] = "No users found for training"
                self.active_tasks[task_id]["end_time"] = datetime.now().isoformat()
                self.active_tasks[task_id]["progress"] = 0.0
                self._save_task_history()
                return

            # Обновляем статус
            self.active_tasks[task_id]["status"] = "training"
            self.active_tasks[task_id]["message"] = f"Training voice model with {len(user_ids)} users"

            # ВАЖНО: Запуск тренировки с callback для обновления прогресса
            def progress_callback(epoch, total_epochs, loss, accuracy):
                """Callback для обновления прогресса во время тренировки"""
                if task_id in self.active_tasks:
                    progress = (epoch / total_epochs) * 100
                    self.active_tasks[task_id]["progress"] = progress
                    self.active_tasks[task_id][
                        "message"] = f"Epoch {epoch}/{total_epochs}, Loss: {loss:.4f}, Acc: {accuracy:.2f}%"
                    logger.info(f"Training progress: {progress:.1f}% - Epoch {epoch}/{total_epochs}")

            # Передаем callback в тренер
            training_result = trainer.train(user_ids, progress_callback=progress_callback)

            # Обработка результата тренировки
            if isinstance(training_result, dict):
                status = training_result.get("status", "error")
                message = training_result.get("message", "Unknown error")
                progress = training_result.get("progress", 0.0)

                self.active_tasks[task_id]["status"] = status
                self.active_tasks[task_id]["progress"] = progress if status != "completed" else 100.0
                self.active_tasks[task_id]["message"] = message

                # Добавляем данные о потерях, если есть
                if "training_loss" in training_result:
                    self.active_tasks[task_id]["training_loss"] = training_result["training_loss"]
                if "best_loss" in training_result:
                    self.active_tasks[task_id]["best_loss"] = training_result["best_loss"]
            else:
                # Обработка булевого результата
                if training_result:
                    self.active_tasks[task_id]["status"] = "completed"
                    self.active_tasks[task_id]["progress"] = 100.0
                    self.active_tasks[task_id]["message"] = "Training completed successfully"
                else:
                    self.active_tasks[task_id]["status"] = "error"
                    self.active_tasks[task_id]["progress"] = 0.0
                    self.active_tasks[task_id]["message"] = "Training failed"

            # Устанавливаем время завершения
            self.active_tasks[task_id]["end_time"] = datetime.now().isoformat()

            # Сохранение в историю
            self.task_history[task_id] = self.active_tasks[task_id].copy()
            self._save_task_history()

            logger.info(
                f"Voice model training task {task_id} finished with status: {self.active_tasks[task_id]['status']}")

        except Exception as e:
            logger.error(f"Error in voice model training thread for task {task_id}: {e}")
            if task_id in self.active_tasks:
                self.active_tasks[task_id]["status"] = "error"
                self.active_tasks[task_id]["progress"] = 0.0
                self.active_tasks[task_id]["message"] = f"Training failed: {str(e)}"
                self.active_tasks[task_id]["end_time"] = datetime.now().isoformat()

                # Сохранение в историю
                self.task_history[task_id] = self.active_tasks[task_id].copy()
                self._save_task_history()

    def _train_anti_spoof_model_thread(self, task_id: str, params: Dict[str, Any]):
        """
        Функция для запуска тренировки модели защиты от спуфинга в отдельном потоке
        """
        try:
            logger.info(f"Starting anti-spoof model training thread for task {task_id}")

            # Создание объекта тренера
            trainer = AntiSpoofTrainer(
                model_path=self.model_path,
                real_audio_path=self.real_voice_path,
                spoof_audio_path=self.spoof_voice_path,
                output_path=os.path.join(self.temp_model_path, task_id),
                device=self.device,
                batch_size=params.get("batch_size", 16),
                learning_rate=params.get("learning_rate", 0.0001),
                num_epochs=params.get("num_epochs", 30)
            )

            # Обновляем статус
            self.active_tasks[task_id]["status"] = "training"
            self.active_tasks[task_id]["message"] = "Training anti-spoofing model"

            # Callback для обновления прогресса
            def progress_callback(epoch, total_epochs, loss, accuracy):
                if task_id in self.active_tasks:
                    progress = (epoch / total_epochs) * 100
                    self.active_tasks[task_id]["progress"] = progress
                    self.active_tasks[task_id][
                        "message"] = f"Anti-spoof: Epoch {epoch}/{total_epochs}, Loss: {loss:.4f}, Acc: {accuracy:.2f}%"
                    logger.info(f"Anti-spoof training progress: {progress:.1f}%")

            # Запуск тренировки с callback
            result = trainer.train(task_id, progress_callback=progress_callback)

            # Обновление статуса задачи
            self.active_tasks[task_id]["status"] = result["status"]
            self.active_tasks[task_id]["progress"] = result.get("progress",
                                                                100.0 if result["status"] == "completed" else 0.0)
            self.active_tasks[task_id]["message"] = result["message"]
            self.active_tasks[task_id]["end_time"] = datetime.now().isoformat()

            # Сохранение в историю
            self.task_history[task_id] = self.active_tasks[task_id].copy()
            self._save_task_history()

            logger.info(f"Anti-spoof model training task {task_id} finished")

        except Exception as e:
            logger.error(f"Error in anti-spoof model training thread for task {task_id}: {e}")
            if task_id in self.active_tasks:
                self.active_tasks[task_id]["status"] = "error"
                self.active_tasks[task_id]["message"] = f"Training failed: {str(e)}"
                self.active_tasks[task_id]["end_time"] = datetime.now().isoformat()
                self.active_tasks[task_id]["progress"] = 0.0

                # Сохранение в историю
                self.task_history[task_id] = self.active_tasks[task_id].copy()
                self._save_task_history()

    def start_voice_model_training(self, params: Dict[str, Any] = None) -> str:
        """
        Запускает тренировку модели распознавания голоса
        """
        if params is None:
            params = {}

        task_id = str(uuid.uuid4())

        # Создание записи о задаче
        self.active_tasks[task_id] = {
            "task_id": task_id,
            "type": "voice_model",
            "status": "starting",
            "progress": 0.0,
            "message": "Initializing voice model training",
            "start_time": datetime.now().isoformat(),
            "end_time": None,
            "params": params,
            "training_loss": [],
            "validation_loss": [],
            "best_loss": None
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
        """
        if params is None:
            params = {}

        task_id = str(uuid.uuid4())

        # Создание записи о задаче
        self.active_tasks[task_id] = {
            "task_id": task_id,
            "type": "anti_spoof",
            "status": "starting",
            "progress": 0.0,
            "message": "Initializing anti-spoofing model training",
            "start_time": datetime.now().isoformat(),
            "end_time": None,
            "params": params,
            "training_loss": [],
            "validation_loss": [],
            "best_loss": None
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
        Возвращает статус задачи тренировки
        """
        try:
            if not task_id or not isinstance(task_id, str):
                logger.warning(f"Invalid task_id: {task_id}")
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

            # Если задача не найдена
            logger.warning(f"Task not found: {task_id}")
            return {
                "task_id": task_id,
                "status": "error",
                "progress": 0.0,
                "message": f"Task with ID {task_id} not found",
                "type": None
            }
        except Exception as e:
            logger.error(f"Error getting training status: {e}")
            return {
                "task_id": task_id if task_id else "unknown",
                "status": "error",
                "progress": 0.0,
                "message": f"Error retrieving task status: {str(e)}",
                "type": None
            }

    def get_all_tasks(self) -> List[Dict[str, Any]]:
        """
        Возвращает список всех задач (активных и из истории)
        """
        all_tasks = []

        # Добавляем активные задачи
        for task in self.active_tasks.values():
            all_tasks.append(task)

        # Добавляем задачи из истории (исключая дубликаты)
        for task_id, task in self.task_history.items():
            if task_id not in self.active_tasks:
                all_tasks.append(task)

        # Сортируем по времени начала (новые сначала)
        all_tasks.sort(key=lambda x: x.get("start_time", ""), reverse=True)

        return all_tasks

    def cancel_task(self, task_id: str) -> bool:
        """
        Отменяет активную задачу тренировки
        """
        try:
            if task_id in self.active_tasks:
                self.active_tasks[task_id]["status"] = "cancelled"
                self.active_tasks[task_id]["message"] = "Training cancelled by user"
                self.active_tasks[task_id]["end_time"] = datetime.now().isoformat()

                # Сохранение в историю
                self.task_history[task_id] = self.active_tasks[task_id].copy()
                self._save_task_history()

                logger.info(f"Task {task_id} cancelled")
                return True
            else:
                logger.warning(f"Cannot cancel task {task_id}: not found in active tasks")
                return False
        except Exception as e:
            logger.error(f"Error cancelling task {task_id}: {e}")
            return False

    def clean_completed_tasks(self, max_age_days: int = 7) -> int:
        """
        Очищает завершенные задачи из истории старше указанного возраста
        """
        try:
            from datetime import datetime, timedelta

            current_time = datetime.now()
            cutoff_time = current_time - timedelta(days=max_age_days)
            tasks_to_remove = []

            logger.info(f"Cleaning tasks older than {max_age_days} days (before {cutoff_time})")

            # Проверяем задачи в истории
            for task_id, task in self.task_history.items():
                try:
                    # Проверяем только завершенные задачи
                    if task.get("status") not in ["completed", "error", "cancelled"]:
                        continue

                    end_time_str = task.get("end_time")
                    if not end_time_str:
                        # Если нет времени окончания, используем время начала
                        end_time_str = task.get("start_time")

                    if end_time_str:
                        try:
                            # Обработка разных форматов времени
                            if 'T' in end_time_str:
                                # ISO format: 2024-05-24T12:30:45.123456
                                if '.' in end_time_str:
                                    end_time_str = end_time_str.split('.')[0]  # Убираем микросекунды
                                task_end_time = datetime.fromisoformat(end_time_str.replace('Z', ''))
                            else:
                                # Другой формат
                                task_end_time = datetime.strptime(end_time_str, "%Y-%m-%d %H:%M:%S")

                            # Проверяем возраст задачи
                            if task_end_time < cutoff_time:
                                tasks_to_remove.append(task_id)
                                logger.info(f"Marking task {task_id} for removal (ended: {task_end_time})")

                        except ValueError as ve:
                            logger.warning(f"Could not parse time for task {task_id}: {end_time_str} - {ve}")
                            # Если не можем парсить время и задача старая, удаляем ее
                            if max_age_days > 30:  # Только если запрашиваем очистку старых задач
                                tasks_to_remove.append(task_id)

                except Exception as te:
                    logger.error(f"Error processing task {task_id}: {te}")
                    continue

            # Удаление задач
            removed_count = 0
            for task_id in tasks_to_remove:
                try:
                    # Удаляем из истории
                    if task_id in self.task_history:
                        del self.task_history[task_id]
                        removed_count += 1
                        logger.info(f"Removed task {task_id} from history")

                    # Также удаляем из активных задач, если есть
                    if task_id in self.active_tasks:
                        del self.active_tasks[task_id]
                        logger.info(f"Removed task {task_id} from active tasks")

                except Exception as de:
                    logger.error(f"Error deleting task {task_id}: {de}")

            # Сохранение обновленной истории
            if removed_count > 0:
                try:
                    self._save_task_history()
                    logger.info(f"Successfully cleaned {removed_count} tasks")
                except Exception as se:
                    logger.error(f"Error saving task history after cleanup: {se}")

            return removed_count

        except Exception as e:
            logger.error(f"Error in clean_completed_tasks: {e}")
            return 0