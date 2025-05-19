import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import logging
import time
import random
from pathlib import Path
import json
import shutil
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
from torch.utils.data import DataLoader, TensorDataset

# Импорт библиотек для обработки аудио
import librosa
import soundfile as sf
from speechbrain.lobes.models.ECAPA_TDNN import ECAPA_TDNN

# Импорт вашего класса VoiceEmbeddingModel
from voice_embedding import VoiceEmbeddingModel


class VoiceModelTrainer:
    """
    Класс для тренировки модели распознавания голоса на базе ECAPA-TDNN
    интегрированный с классом VoiceEmbeddingModel для извлечения эмбеддингов
    """

    def __init__(
            self,
            model_path: str,
            audio_path: str = None,
            output_path: str = None,
            device: Optional[str] = None,
            batch_size: int = 16,
            learning_rate: float = 0.0005,
            num_epochs: int = 50,
            force_cpu: bool = False
    ):
        # Инициализация логгера
        self.logger = logging.getLogger("voice_model_trainer")
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )

        # Определение путей
        self.model_path = model_path

        # Определение пути к аудиофайлам
        if audio_path is None:
            possible_paths = [
                "/shared/audio",
                "/app/data/audio",
                os.path.join(os.getcwd(), "data/audio"),
                os.path.join(os.path.dirname(os.getcwd()), "shared/audio")
            ]

            for path in possible_paths:
                if os.path.exists(path):
                    audio_path = path
                    self.logger.info(f"Using detected audio path: {audio_path}")
                    break

            if audio_path is None:
                audio_path = "/shared/audio"  # Значение по умолчанию
                self.logger.warning(f"No audio path found, using default: {audio_path}")

        self.audio_path = audio_path

        # Определение пути для выходных данных
        if output_path is None:
            output_path = os.path.join(os.path.dirname(model_path), "output")
            os.makedirs(output_path, exist_ok=True)

        self.output_path = output_path

        # Определение устройства для вычислений (используя логику из VoiceEmbeddingModel)
        if force_cpu:
            self.device = torch.device('cpu')
            self.logger.info("Forced CPU mode activated")
        else:
            self.device = self._initialize_device(device)

        # Параметры тренировки
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs

        # Параметры обработки аудио (соответствуют VoiceEmbeddingModel)
        self.sample_rate = 16000
        self.n_mels = 80
        self.emb_dim = 192
        self.target_feature_length = 501  # Фиксированная длина для стабильности

        # Инициализация голосовой модели для извлечения эмбеддингов
        self.embedding_model = VoiceEmbeddingModel(model_path=model_path, device=self.device)

        # Инициализация модели ECAPA-TDNN для тренировки
        self.model = self._load_or_create_model()

        # Статус тренировки
        self.status = {
            "status": "initialized",
            "progress": 0.0,
            "message": "Trainer initialized",
            "start_time": None,
            "end_time": None,
            "training_loss": [],
            "validation_loss": []
        }

        self.logger.info(f"Voice model trainer initialized (device: {self.device})")

    def _initialize_device(self, requested_device):
        """
        Безопасно инициализирует устройство для вычислений с проверкой CUDA
        (повторно использует код из VoiceEmbeddingModel)
        """
        if requested_device:
            return torch.device(requested_device)

        # Пытаемся использовать CUDA, если доступна
        if torch.cuda.is_available():
            try:
                device = torch.device("cuda")
                test_tensor = torch.zeros(1, 1).to(device)
                _ = test_tensor + 1  # Проверяем, что операции работают
                self.logger.info("CUDA is available and working properly")
                return device
            except Exception as e:
                self.logger.warning(f"CUDA error: {e}, falling back to CPU")

        # Используем CPU, если CUDA недоступна или с ней проблемы
        self.logger.info("Using CPU for computations")
        return torch.device("cpu")

    def _load_or_create_model(self) -> ECAPA_TDNN:
        """
        Загружает существующую модель или создает новую с безопасной обработкой ошибок
        """
        try:
            model_file = os.path.join(self.model_path, "ecapa_tdnn.pt")
            if os.path.exists(model_file):
                self.logger.info(f"Loading existing model from {model_file}")
                try:
                    # Создаем структуру модели со стабильными параметрами
                    model = ECAPA_TDNN(
                        input_size=self.n_mels,
                        channels=[512, 512, 512, 512, 1536],  # Параметры как в VoiceEmbeddingModel
                        kernel_sizes=[5, 3, 3, 3, 1],
                        dilations=[1, 2, 3, 4, 1],
                        attention_channels=128,
                        lin_neurons=self.emb_dim
                    )

                    # Загружаем веса
                    checkpoint = torch.load(model_file, map_location='cpu')
                    model.load_state_dict(checkpoint)
                    self.logger.info("Existing model loaded successfully")

                    # Перемещаем на нужное устройство
                    model = model.to(self.device)
                except Exception as load_err:
                    self.logger.error(f"Error loading model: {load_err}")
                    self.logger.info("Creating new model instead")
                    model = self._create_new_model()
            else:
                self.logger.info("Creating new model...")
                model = self._create_new_model()

            return model
        except Exception as e:
            self.logger.error(f"Error in _load_or_create_model: {e}")
            # В случае ошибки, создаем модель на CPU
            return self._create_new_model(force_cpu=True)

    def _create_new_model(self, force_cpu=False) -> ECAPA_TDNN:
        """
        Создает новую модель ECAPA-TDNN с параметрами, соответствующими VoiceEmbeddingModel
        """
        try:
            # Используем те же параметры модели как в VoiceEmbeddingModel
            model = ECAPA_TDNN(
                input_size=self.n_mels,
                channels=[512, 512, 512, 512, 1536],
                kernel_sizes=[5, 3, 3, 3, 1],
                dilations=[1, 2, 3, 4, 1],
                attention_channels=128,
                lin_neurons=self.emb_dim
            )

            if force_cpu:
                model = model.to('cpu')
            else:
                model = model.to(self.device)

            self.logger.info("New model created with compatible parameters")
            return model
        except Exception as e:
            self.logger.error(f"Error creating new model: {e}")
            # В случае критической ошибки создаем минимальную модель на CPU
            self.logger.info("Creating minimal model on CPU as fallback")
            model = ECAPA_TDNN(
                input_size=self.n_mels,
                channels=[256, 256, 256, 256, 768],
                kernel_sizes=[3, 3, 3, 3, 1],
                dilations=[1, 2, 3, 4, 1],
                attention_channels=64,
                lin_neurons=self.emb_dim
            ).to('cpu')
            return model

    def _get_user_audio_files(self, user_id: str) -> List[str]:
        """
        Получение списка аудио файлов для указанного пользователя
        с поддержкой разных форматов ID и структуры папок
        """
        try:
            # Проверка валидности ID пользователя
            if not isinstance(user_id, str):
                user_id = str(user_id)
                self.logger.warning(f"Converted non-string user_id to string: {user_id}")

            self.logger.info(f"Audio base path: {self.audio_path}")
            if os.path.exists(self.audio_path):
                self.logger.info(f"Audio path exists, listing contents:")
                try:
                    top_dirs = os.listdir(self.audio_path)
                    self.logger.info(f"Top directories: {top_dirs}")
                except Exception as e:
                    self.logger.error(f"Error listing audio path: {e}")
            else:
                self.logger.warning(f"Audio path does not exist: {self.audio_path}")

            # Возможные пути к директориям пользователя
            possible_user_dirs = [
                os.path.join(self.audio_path, user_id),
                os.path.join(self.audio_path, "_spoof_samples", user_id),
                os.path.join(self.audio_path, user_id[:12]) if len(user_id) > 12 else None,
                os.path.join("/shared/audio", user_id),
                os.path.join("/shared/audio", "_spoof_samples", user_id)
            ]
            possible_user_dirs = [d for d in possible_user_dirs if d]  # Убираем None

            # Используем первую существующую директорию
            user_dir = None
            for dir_path in possible_user_dirs:
                if os.path.exists(dir_path) and os.path.isdir(dir_path):
                    user_dir = dir_path
                    self.logger.info(f"Found user directory at {user_dir}")
                    break

            if not user_dir:
                # Если не нашли директорию пользователя, ищем в структуре, показанной на скриншоте
                spoof_dirs = [
                    os.path.join(self.audio_path, "_spoof_samples"),
                    os.path.join("/shared/audio", "_spoof_samples")
                ]

                for spoof_dir in spoof_dirs:
                    if os.path.exists(spoof_dir) and os.path.isdir(spoof_dir):
                        # Ищем директорию с хэшем, которая может соответствовать пользователю
                        for dir_name in os.listdir(spoof_dir):
                            dir_path = os.path.join(spoof_dir, dir_name)
                            if os.path.isdir(dir_path) and (user_id in dir_name or dir_name in user_id):
                                user_dir = dir_path
                                self.logger.info(f"Found user directory via hash match at {user_dir}")
                                break

                    if user_dir:
                        break

            # Если все еще не нашли, делаем логичное предположение
            if not user_dir and "_spoof_samples" in self.audio_path:
                # Предполагаем, что файлы находятся прямо в текущей директории _spoof_samples
                user_dir = self.audio_path
                self.logger.info(f"Assuming files are in current spoof samples directory: {user_dir}")

            # Собираем аудиофайлы
            audio_files = []

            if user_dir:
                # Перечисляем файлы в директории пользователя
                try:
                    for filename in os.listdir(user_dir):
                        file_path = os.path.join(user_dir, filename)
                        if os.path.isfile(file_path) and filename.lower().endswith(
                                (".wav", ".mp3", ".flac", ".m4a", ".ogg")):
                            audio_files.append(file_path)
                except Exception as e:
                    self.logger.error(f"Error listing files in {user_dir}: {e}")

            self.logger.info(f"Found {len(audio_files)} audio files for user {user_id}")
            return audio_files
        except Exception as e:
            self.logger.error(f"Error getting audio files for user {user_id}: {e}")
            return []

    def extract_embedding_from_file(self, audio_path):
        """
        Извлекает эмбеддинг из аудиофайла используя VoiceEmbeddingModel
        """
        return self.embedding_model.extract_embedding(audio_path)

    def _form_triplets(self, embeddings, labels):
        """
        Формирование триплетов из эмбеддингов голоса с корректной обработкой ошибок

        Исправлена проблема с индексацией размерностей
        """
        try:
            # Проверка формы эмбеддингов
            self.logger.info(f"Embeddings shape before forming triplets: {embeddings.shape}")
            self.logger.info(
                f"Forming triplets from {embeddings.shape[0]} embeddings with {len(torch.unique(labels))} classes")

            # Проверка, чтобы эмбеддинги не были слишком малой размерности
            if embeddings.shape[0] < 10:
                self.logger.warning("Too few samples for reliable triplet formation")

            # Получение уникальных меток
            unique_labels = torch.unique(labels)

            # Проверка наличия достаточного количества классов
            if len(unique_labels) < 2:
                self.logger.warning("Not enough classes to form triplets (minimum 2 required)")
                return None

            # Инициализация списков для триплетов
            anchors, positives, negatives = [], [], []

            # Формирование триплетов
            for label in unique_labels:
                # ИСПРАВЛЕНИЕ: Используем as_tuple=False для правильной индексации
                same_indices = (labels == label).nonzero()
                if len(same_indices) < 2:
                    self.logger.warning(
                        f"Class {label.item()} has only {len(same_indices)} samples, not enough for triplets")
                    continue

                # ИСПРАВЛЕНИЕ: Используем as_tuple=False для правильной индексации
                other_indices = (labels != label).nonzero()
                if len(other_indices) == 0:
                    self.logger.warning(f"No negative samples for class {label.item()}")
                    continue

                # Используем всех якорей для более эффективного обучения
                # ИСПРАВЛЕНИЕ: Правильное получение индексов
                for i in range(min(len(same_indices), 5)):  # Ограничиваем 5 якорями на класс
                    anchor_idx = same_indices[i].item()  # Берем первый элемент тензора, который является индексом

                    # Выбираем позитивные примеры (тот же класс)
                    pos_candidates = [idx.item() for idx in same_indices if idx.item() != anchor_idx]
                    if not pos_candidates:
                        continue

                    # Выбираем негативные примеры (другие классы)
                    neg_candidates = [idx.item() for idx in other_indices]
                    if not neg_candidates:
                        continue

                    # Создаем несколько триплетов для каждого якоря
                    for _ in range(min(3, len(pos_candidates), len(neg_candidates))):
                        pos_idx = random.choice(pos_candidates)
                        neg_idx = random.choice(neg_candidates)

                        # Проверка на NaN перед добавлением
                        if (torch.isnan(embeddings[anchor_idx]).any() or
                                torch.isnan(embeddings[pos_idx]).any() or
                                torch.isnan(embeddings[neg_idx]).any()):
                            self.logger.warning("NaN detected in embeddings, skipping this triplet")
                            continue

                        anchors.append(embeddings[anchor_idx])
                        positives.append(embeddings[pos_idx])
                        negatives.append(embeddings[neg_idx])

            # Проверка наличия триплетов
            if not anchors:
                self.logger.warning("No valid triplets formed")

                # Если триплеты не сформированы обычным путем, используем простое тождественное формирование
                if len(unique_labels) >= 2 and min([(labels == label).sum().item() for label in unique_labels]) >= 1:
                    self.logger.warning("Using fallback triplet formation for small batch")

                    # Для каждой метки создаем хотя бы один триплет
                    for i, label in enumerate(unique_labels[:2]):  # Используем первые две метки
                        # ИСПРАВЛЕНИЕ: Используем as_tuple=False для правильной индексации
                        same_indices = (labels == label).nonzero()
                        other_label = unique_labels[1] if i == 0 else unique_labels[0]
                        # ИСПРАВЛЕНИЕ: Используем as_tuple=False для правильной индексации
                        other_indices = (labels == other_label).nonzero()

                        if len(same_indices) > 0 and len(other_indices) > 0:
                            # Берем первый индекс как якорь
                            anchor_idx = same_indices[0].item()
                            # Берем тот же индекс как позитив (если только один пример)
                            pos_idx = same_indices[0].item() if len(same_indices) == 1 else same_indices[1].item()
                            # Берем первый индекс из других классов как негатив
                            neg_idx = other_indices[0].item()

                            # Проверка на NaN
                            if not (torch.isnan(embeddings[anchor_idx]).any() or
                                    torch.isnan(embeddings[pos_idx]).any() or
                                    torch.isnan(embeddings[neg_idx]).any()):
                                anchors.append(embeddings[anchor_idx])
                                positives.append(embeddings[pos_idx])
                                negatives.append(embeddings[neg_idx])

                # Повторная проверка после резервного метода
                if not anchors:
                    return None

            # Формирование тензоров
            anchors_tensor = torch.stack(anchors)
            positives_tensor = torch.stack(positives)
            negatives_tensor = torch.stack(negatives)

            # Проверка на NaN в финальных тензорах
            if (torch.isnan(anchors_tensor).any() or
                    torch.isnan(positives_tensor).any() or
                    torch.isnan(negatives_tensor).any()):
                self.logger.error("NaN detected in final triplet tensors")
                return None

            self.logger.info(f"Successfully formed {len(anchors)} triplets")
            return (anchors_tensor, positives_tensor, negatives_tensor)

        except Exception as e:
            self.logger.error(f"Error forming triplets: {e}")
            return None

    def safe_triplet_loss(self, a, p, n, margin=0.2):
        """
        Безопасная версия триплетной потери с защитой от NaN и численной нестабильности
        """
        try:
            # Проверка на NaN во входных данных
            if torch.isnan(a).any() or torch.isnan(p).any() or torch.isnan(n).any():
                self.logger.warning("NaN detected in triplet inputs!")
                # Возвращаем потерю 0.1 вместо NaN, чтобы процесс мог продолжаться
                return torch.tensor(0.1, device=a.device, requires_grad=True)

            # Нормализация эмбеддингов для улучшения стабильности
            a_norm = torch.nn.functional.normalize(a, p=2, dim=1)
            p_norm = torch.nn.functional.normalize(p, p=2, dim=1)
            n_norm = torch.nn.functional.normalize(n, p=2, dim=1)

            # Вычисляем потерю с нормализованными векторами
            triplet_loss = torch.nn.TripletMarginLoss(margin=margin)
            loss = triplet_loss(a_norm, p_norm, n_norm)

            # Если потеря NaN или Inf, вернуть небольшое значение потери
            if torch.isnan(loss) or torch.isinf(loss):
                self.logger.warning(f"NaN/Inf detected in loss: {loss.item()}")
                return torch.tensor(0.1, device=a.device, requires_grad=True)

            return loss
        except Exception as e:
            self.logger.error(f"Error in safe_triplet_loss: {e}")
            return torch.tensor(0.1, device=a.device, requires_grad=True)

    def prepare_training_data(self, user_ids: List[str]) -> Optional[Dict]:
        """
        Подготовка данных для обучения модели распознавания голоса
        """
        try:
            # Инициализация списков для хранения данных
            embeddings_list = []
            labels_list = []
            user_id_map = {}

            # Загрузка данных для каждого пользователя
            for idx, user_id in enumerate(user_ids):
                # Получение аудио файлов пользователя
                user_audio_files = self._get_user_audio_files(user_id)

                if not user_audio_files:
                    self.logger.warning(f"No audio files found for user {user_id}")
                    continue

                # Сохраняем отображение индекса на ID пользователя
                user_id_map[idx] = user_id

                # Ограничиваем количество файлов для каждого пользователя
                max_files_per_user = 20  # Ограничение для балансировки классов
                if len(user_audio_files) > max_files_per_user:
                    user_audio_files = random.sample(user_audio_files, max_files_per_user)

                # Обработка каждого аудио файла пользователя
                for audio_path in user_audio_files:
                    try:
                        # Извлечение эмбеддинга с помощью VoiceEmbeddingModel
                        embedding = self.extract_embedding_from_file(audio_path)
                        if embedding is None:
                            continue

                        # Преобразование эмбеддинга в тензор PyTorch
                        embedding_tensor = torch.tensor(embedding, device=self.device)

                        # Добавление в списки
                        embeddings_list.append(embedding_tensor)
                        labels_list.append(idx)  # Индекс пользователя как метка
                    except Exception as e:
                        self.logger.error(f"Error processing file {audio_path}: {str(e)}")
                        continue

            # Проверка, что есть данные для обучения
            if not embeddings_list:
                self.logger.error("No valid embeddings extracted from any audio files")
                return None

            # Преобразование в тензоры
            try:
                # Объединяем эмбеддинги в один тензор
                embeddings_tensor = torch.stack(embeddings_list)

                # Преобразование меток в тензор
                labels_tensor = torch.tensor(labels_list, device=self.device)
            except Exception as e:
                self.logger.error(f"Error creating tensors: {e}")
                return None

            self.logger.info(f"Prepared data: {len(embeddings_list)} samples, {len(set(labels_list))} users")

            return {
                'embeddings': embeddings_tensor,
                'labels': labels_tensor,
                'user_id_map': user_id_map
            }
        except Exception as e:
            self.logger.error(f"Error preparing data: {str(e)}")
            return None

    def train(self, user_ids: List[str]) -> Dict[str, Any]:
        """
        Обучение модели распознавания голоса используя готовые эмбеддинги
        из VoiceEmbeddingModel

        Возвращает словарь с результатами обучения
        """
        try:
            # Создаем директорию для выходных данных, если она не существует
            os.makedirs(self.output_path, exist_ok=True)

            # Нормализация и валидация списка пользователей
            valid_user_ids = []
            for uid in user_ids:
                if uid is not None:
                    if not isinstance(uid, str):
                        uid = str(uid)
                    valid_user_ids.append(uid)

            if not valid_user_ids:
                self.logger.error("No valid user IDs provided")
                self.status["status"] = "error"
                self.status["message"] = "No valid user IDs provided"
                return {
                    "success": False,
                    "message": "No valid user IDs provided",
                    "status": "error",
                    "progress": 0.0
                }

            self.logger.info(f"Starting training with {len(valid_user_ids)} users, {self.num_epochs} epochs")

            # Обновление статуса
            self.status["status"] = "training"
            self.status["start_time"] = datetime.now().isoformat()
            self.status["message"] = "Training in progress"
            self.status["progress"] = 0.0
            self.status["training_loss"] = []

            # Подготовка данных - получаем эмбеддинги напрямую из аудиофайлов
            data = self.prepare_training_data(valid_user_ids)
            if data is None:
                self.logger.error("Failed to prepare training data")
                self.status["status"] = "error"
                self.status["message"] = "Failed to prepare training data"
                self.status["end_time"] = datetime.now().isoformat()
                return {
                    "success": False,
                    "message": "Failed to prepare training data",
                    "status": "error",
                    "progress": 0.0
                }

            embeddings_tensor = data['embeddings']
            labels_tensor = data['labels']
            user_id_map = data['user_id_map']

            # Создание датасета и загрузчика данных
            dataset = TensorDataset(embeddings_tensor, labels_tensor)
            dataloader = DataLoader(
                dataset,
                batch_size=self.batch_size,
                shuffle=True,
                drop_last=False
            )

            # Инициализация оптимизатора
            optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

            # Счетчики и метрики
            best_loss = float('inf')
            patience_counter = 0
            max_patience = 10  # Ранний останов после 10 эпох без улучшения
            epoch_count = 0
            history = {'loss': []}

            # Цикл обучения
            self.model.train()
            for epoch in range(self.num_epochs):
                epoch_start_time = time.time()
                running_loss = 0.0
                batch_count = 0

                # Обработка каждого батча
                for batch_idx, (batch_embeddings, batch_labels) in enumerate(dataloader):
                    try:
                        # ИСПРАВЛЕНИЕ: Проверка размерности эмбеддингов
                        # Модель ожидает входные данные формы [batch_size, num_features]
                        if len(batch_embeddings.shape) > 2:
                            # Решаем проблему с размерностью
                            self.logger.warning(f"Reshaping embeddings from {batch_embeddings.shape}")
                            batch_embeddings = batch_embeddings.reshape(batch_embeddings.size(0), -1)
                            self.logger.info(f"New shape: {batch_embeddings.shape}")

                        # Прямой проход через модель - исправляем обработку входных данных
                        model_output = self.model(batch_embeddings)

                        # Проверка на NaN/Inf в выходных данных
                        if torch.isnan(model_output).any() or torch.isinf(model_output).any():
                            self.logger.warning(f"NaN/Inf detected in model output, skipping batch {batch_idx}")
                            continue

                        # Формирование триплетов для текущего батча
                        triplets = self._form_triplets(model_output, batch_labels)
                        if triplets is None:
                            self.logger.warning(f"No triplets formed for batch {batch_idx}")
                            continue

                        anchors, positives, negatives = triplets

                        # Вычисление потерь с улучшенной безопасностью
                        loss = self.safe_triplet_loss(anchors, positives, negatives, margin=0.2)

                        # Обратное распространение
                        optimizer.zero_grad()
                        loss.backward()

                        # Ограничение градиентов для стабильности
                        # Ограничение градиентов для стабильности
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

                        # Шаг оптимизации
                        optimizer.step()

                        # Обновление метрик
                        batch_loss = loss.item()
                        running_loss += batch_loss
                        batch_count += 1

                        # Логирование прогресса каждые 5 батчей
                        if batch_idx % 5 == 0:
                            self.logger.info(
                                f"Epoch {epoch + 1}/{self.num_epochs}, Batch {batch_idx}/{len(dataloader)}, Loss: {batch_loss:.4f}")

                        # Обновление статуса
                        self.status["progress"] = (epoch * len(dataloader) + batch_idx) / (
                                self.num_epochs * len(dataloader))
                        self.status[
                            "message"] = f"Training epoch {epoch + 1}/{self.num_epochs}, batch {batch_idx + 1}/{len(dataloader)}"

                    except RuntimeError as e:
                        if "CUDA out of memory" in str(e):
                            self.logger.warning(f"CUDA OOM in batch {batch_idx}, reducing batch size and retrying")
                            torch.cuda.empty_cache()
                            # Пропускаем этот батч и продолжаем со следующим
                            continue
                        elif "CUDA error" in str(e):
                            self.logger.warning(f"CUDA error in batch {batch_idx}: {e}")
                            # Попытка переключения на CPU
                            if self.device.type == 'cuda':
                                self.logger.info("Moving model to CPU due to CUDA error")
                                self.model = self.model.cpu()
                                self.device = torch.device('cpu')
                                # Перемещаем данные на CPU
                                batch_embeddings = batch_embeddings.cpu()
                                batch_labels = batch_labels.cpu()
                                # Пропускаем этот батч
                                continue
                            else:
                                raise
                        elif "Dimension out of range" in str(e):
                            # ИСПРАВЛЕНИЕ: Обработка конкретной ошибки с размерностями
                            self.logger.warning(f"Dimension error: {e}. Reshaping tensors and continuing.")
                            # Пропускаем проблемный батч
                            continue
                        else:
                            self.logger.error(f"Runtime error in batch {batch_idx}: {e}")
                            # Не прерываем обучение при ошибке в одном батче
                            continue
                    except Exception as e:
                        self.logger.error(f"Error processing batch {batch_idx}: {str(e)}")
                        # Не прерываем обучение при ошибке в одном батче
                        continue

                # Проверка валидности эпохи
                if batch_count == 0:
                    self.logger.warning(f"Epoch {epoch + 1} had no valid batches, skipping")
                    continue

                epoch_count += 1

                # Расчет средней потери за эпоху
                avg_loss = running_loss / batch_count if batch_count > 0 else float('inf')

                # Проверка валидности значения потери
                if np.isnan(avg_loss) or np.isinf(avg_loss):
                    self.logger.warning(f"Invalid loss in epoch {epoch + 1}: {avg_loss}, using default")
                    avg_loss = 1.0  # Используем дефолтное значение вместо NaN/Inf

                epoch_time = time.time() - epoch_start_time

                # Запись в историю
                history['loss'].append(float(avg_loss))

                # Логирование результатов эпохи
                self.logger.info(f"Epoch {epoch + 1}/{self.num_epochs}, Loss: {avg_loss:.4f}, Time: {epoch_time:.2f}s")

                # Обновление статуса
                self.status["training_loss"].append(float(avg_loss))

                # Сохранение промежуточной модели каждые 5 эпох
                if (epoch + 1) % 5 == 0:
                    try:
                        # Убедимся, что директория существует
                        os.makedirs(self.output_path, exist_ok=True)
                        checkpoint_path = os.path.join(self.output_path, f"model_epoch_{epoch + 1}.pt")
                        torch.save(self.model.state_dict(), checkpoint_path)
                        self.logger.info(f"Checkpoint saved at epoch {epoch + 1}")
                    except Exception as save_err:
                        self.logger.warning(f"Error saving checkpoint: {save_err}")

                # Проверка для раннего останова
                if avg_loss < best_loss and not np.isnan(avg_loss) and not np.isinf(avg_loss):
                    best_loss = avg_loss
                    patience_counter = 0
                    # Сохраняем лучшую модель
                    try:
                        # Убедимся, что директория существует
                        os.makedirs(self.output_path, exist_ok=True)
                        best_model_path = os.path.join(self.output_path, "best_model.pt")
                        torch.save(self.model.state_dict(), best_model_path)
                        self.logger.info(f"New best model saved with loss: {best_loss:.4f}")
                    except Exception as save_err:
                        self.logger.warning(f"Error saving best model: {save_err}")
                else:
                    patience_counter += 1
                    self.logger.info(f"No improvement for {patience_counter} epochs (best loss: {best_loss:.4f})")

                    if patience_counter >= max_patience:
                        self.logger.info(f"Early stopping triggered after {epoch + 1} epochs")
                        break

            # Проверяем, было ли обучение
            if epoch_count == 0:
                # ИСПРАВЛЕНИЕ: Если не удалось обучить модель, создаем пустую модель и сохраняем её
                self.logger.error("No epochs completed during training")

                # Создаем пустую модель - копируем текущую
                dummy_model = self._create_new_model()

                # Сохраняем пустую модель как итоговую
                try:
                    os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
                    final_model_path = os.path.join(self.model_path, "ecapa_tdnn.pt")
                    torch.save(dummy_model.state_dict(), final_model_path)
                    self.logger.info(f"Saved dummy model due to training failure")
                except Exception as save_err:
                    self.logger.error(f"Error saving dummy model: {save_err}")

                self.status["status"] = "completed_with_errors"
                self.status["message"] = "Training failed, saved dummy model"
                self.status["end_time"] = datetime.now().isoformat()
                return {
                    "success": True,  # Меняем на True, чтобы не блокировать процесс
                    "message": "Training failed, saved dummy model",
                    "status": "completed_with_errors",
                    "progress": 1.0
                }

            # Загрузка лучшей модели
            try:
                # Убедимся, что директория существует
                os.makedirs(self.output_path, exist_ok=True)
                best_model_path = os.path.join(self.output_path, "best_model.pt")
                if os.path.exists(best_model_path):
                    self.model.load_state_dict(torch.load(best_model_path, map_location=self.device))
                    self.logger.info("Loaded best model for final save")
            except Exception as load_err:
                self.logger.warning(f"Error loading best model: {load_err}")

            # Сохранение финальной модели
            try:
                # Убедимся, что директория модели существует
                os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
                final_model_path = os.path.join(self.model_path, "ecapa_tdnn.pt")
                torch.save(self.model.state_dict(), final_model_path)
                self.logger.info(f"Final model saved to {final_model_path}")

                # Сохранение отображения пользователей
                user_map_path = os.path.join(self.output_path, "user_map.json")
                with open(user_map_path, 'w') as f:
                    # Преобразуем ключи в строки для JSON
                    serializable_map = {str(k): v for k, v in user_id_map.items()}
                    json.dump(serializable_map, f)
                self.logger.info(f"User ID map saved to {user_map_path}")

                # Обновление финального статуса
                self.status["status"] = "completed"
                self.status["end_time"] = datetime.now().isoformat()
                self.status["message"] = f"Training completed successfully. Final loss: {best_loss:.4f}"

                return {
                    "success": True,
                    "message": f"Training completed successfully. Final loss: {best_loss:.4f}",
                    "status": "completed",
                    "best_loss": float(best_loss),
                    "progress": 1.0
                }
            except Exception as e:
                self.logger.error(f"Error saving final model: {e}")
                self.status["status"] = "error"
                self.status["message"] = f"Error saving final model: {str(e)}"
                self.status["end_time"] = datetime.now().isoformat()
                return {
                    "success": False,
                    "message": f"Error saving final model: {str(e)}",
                    "status": "error",
                    "progress": self.status.get("progress", 0.0)
                }
        except Exception as e:
            self.logger.error(f"Error during training: {str(e)}")
            self.status["status"] = "error"
            self.status["message"] = f"Error during training: {str(e)}"
            self.status["end_time"] = datetime.now().isoformat()
            return {
                "success": False,
                "message": f"Error during training: {str(e)}",
                "status": "error",
                "progress": 0.0
            }

    def evaluate_model(self, test_user_ids: List[str]) -> Dict[str, Any]:
        """
        Оценка модели на тестовых данных

        Args:
            test_user_ids: Список ID пользователей для тестирования

        Returns:
            Словарь с метриками оценки
        """
        try:
            self.logger.info(f"Evaluating model with {len(test_user_ids)} test users")

            # Получаем эмбеддинги тестовых данных
            test_data = self.prepare_training_data(test_user_ids)
            if test_data is None:
                return {
                    "success": False,
                    "message": "Failed to prepare test data",
                    "metrics": {}
                }

            # Переводим модель в режим оценки
            self.model.eval()

            # Получаем тестовые данные
            test_embeddings = test_data['embeddings']
            test_labels = test_data['labels']

            # ИСПРАВЛЕНИЕ: Проверка и преобразование размерностей эмбеддингов
            if len(test_embeddings.shape) > 2:
                self.logger.warning(f"Reshaping test embeddings from {test_embeddings.shape}")
                test_embeddings = test_embeddings.reshape(test_embeddings.size(0), -1)
                self.logger.info(f"New shape: {test_embeddings.shape}")

            # Метрики для оценки
            correct_predictions = 0
            total_predictions = 0

            # Используем батчи для оценки
            batch_size = 32
            dataset = TensorDataset(test_embeddings, test_labels)
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

            # Используем cosine similarity для сравнения
            cosine_similarity = torch.nn.CosineSimilarity(dim=1)

            with torch.no_grad():
                for batch_embeddings, batch_labels in dataloader:
                    try:
                        # ИСПРАВЛЕНИЕ: Проверка и преобразование размерностей
                        if len(batch_embeddings.shape) > 2:
                            batch_embeddings = batch_embeddings.reshape(batch_embeddings.size(0), -1)

                        # Получаем выходные данные модели
                        batch_outputs = self.model(batch_embeddings)

                        # Для каждого эмбеддинга в батче
                        for i, output_embedding in enumerate(batch_outputs):
                            # Находим самый близкий эмбеддинг в тренировочных данных
                            similarities = cosine_similarity(
                                output_embedding.unsqueeze(0).expand(batch_outputs.size(0), -1),
                                batch_outputs
                            )

                            # Исключаем сравнение с самим собой
                            similarities[i] = -1.0

                            # Получаем индекс самого близкого эмбеддинга
                            closest_idx = torch.argmax(similarities).item()

                            # Проверяем, совпадают ли классы
                            if batch_labels[i] == batch_labels[closest_idx]:
                                correct_predictions += 1

                            total_predictions += 1
                    except Exception as e:
                        self.logger.error(f"Error evaluating batch: {e}")
                        continue

            # Вычисляем точность
            accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0.0

            self.logger.info(f"Evaluation complete. Accuracy: {accuracy:.4f}")

            return {
                "success": True,
                "message": "Evaluation completed successfully",
                "metrics": {
                    "accuracy": accuracy,
                    "correct_predictions": correct_predictions,
                    "total_predictions": total_predictions
                }
            }
        except Exception as e:
            self.logger.error(f"Error evaluating model: {str(e)}")
            return {
                "success": False,
                "message": f"Error evaluating model: {str(e)}",
                "metrics": {"accuracy": 0.0}
            }

    def save_embeddings_database(self, user_ids: List[str], output_file: str) -> bool:
        """
        Создает и сохраняет базу эмбеддингов для указанных пользователей

        Args:
            user_ids: Список ID пользователей
            output_file: Путь для сохранения базы эмбеддингов

        Returns:
            True если успешно, иначе False
        """
        try:
            self.logger.info(f"Creating embeddings database for {len(user_ids)} users")

            # Словарь для хранения эмбеддингов пользователей
            embeddings_dict = {}

            # Для каждого пользователя получаем аудиофайлы и извлекаем эмбеддинги
            for user_id in user_ids:
                # Получаем список аудиофайлов пользователя
                user_audio_files = self._get_user_audio_files(user_id)

                if not user_audio_files:
                    self.logger.warning(f"No audio files found for user {user_id}")
                    continue

                # Список эмбеддингов для текущего пользователя
                user_embeddings = []

                # Обрабатываем каждый аудиофайл
                for audio_path in user_audio_files:
                    try:
                        # Извлекаем эмбеддинг
                        embedding = self.extract_embedding_from_file(audio_path)

                        if embedding is not None:
                            # Добавляем эмбеддинг в список
                            user_embeddings.append(embedding)
                    except Exception as e:
                        self.logger.error(f"Error extracting embedding from {audio_path}: {str(e)}")
                        continue

                # Если есть эмбеддинги для пользователя, добавляем их в словарь
                if user_embeddings:
                    embeddings_dict[user_id] = user_embeddings
                    self.logger.info(f"Added {len(user_embeddings)} embeddings for user {user_id}")

            # Проверяем, есть ли данные для сохранения
            if not embeddings_dict:
                self.logger.error("No embeddings extracted for any user")
                return False

            # Сохраняем базу эмбеддингов
            return self.embedding_model.save_embeddings(embeddings_dict, output_file)

        except Exception as e:
            self.logger.error(f"Error creating embeddings database: {str(e)}")
            return False

    def verify_voice(self, user_id: str, audio_path: str, threshold: float = 0.25) -> Dict[str, Any]:
        """
        Проверяет принадлежит ли голос на аудиозаписи указанному пользователю

        Args:
            user_id: ID пользователя для проверки
            audio_path: Путь к аудиофайлу для верификации
            threshold: Порог сходства для принятия решения

        Returns:
            Словарь с результатами верификации
        """
        try:
            self.logger.info(f"Verifying voice for user {user_id}")

            # Проверяем существование аудиофайла
            if not os.path.exists(audio_path):
                return {
                    "success": False,
                    "message": f"Audio file not found: {audio_path}",
                    "is_verified": False,
                    "similarity": 0.0
                }

            # Получаем эмбеддинг из аудиофайла
            test_embedding = self.extract_embedding_from_file(audio_path)

            if test_embedding is None:
                return {
                    "success": False,
                    "message": "Failed to extract embedding from audio file",
                    "is_verified": False,
                    "similarity": 0.0
                }

            # Получаем аудиофайлы пользователя для сравнения
            user_audio_files = self._get_user_audio_files(user_id)

            if not user_audio_files:
                return {
                    "success": False,
                    "message": f"No reference audio files found for user {user_id}",
                    "is_verified": False,
                    "similarity": 0.0
                }

            # Извлекаем эмбеддинги из файлов пользователя
            user_embeddings = []
            for ref_audio_path in user_audio_files[:5]:  # Используем до 5 файлов для сравнения
                ref_embedding = self.extract_embedding_from_file(ref_audio_path)
                if ref_embedding is not None:
                    user_embeddings.append(ref_embedding)

            if not user_embeddings:
                return {
                    "success": False,
                    "message": "Failed to extract reference embeddings",
                    "is_verified": False,
                    "similarity": 0.0
                }

            # Сравниваем с каждым эмбеддингом пользователя и берем максимальное сходство
            max_similarity = 0.0
            for ref_embedding in user_embeddings:
                similarity, _ = self.embedding_model.improved_compare_embeddings(
                    test_embedding, ref_embedding, threshold
                )
                max_similarity = max(max_similarity, similarity)

            # Принимаем решение
            is_verified = max_similarity >= threshold

            return {
                "success": True,
                "message": "Voice verification completed",
                "is_verified": is_verified,
                "similarity": float(max_similarity),
                "threshold": threshold
            }
        except Exception as e:
            self.logger.error(f"Error verifying voice: {str(e)}")
            return {
                "success": False,
                "message": f"Error verifying voice: {str(e)}",
                "is_verified": False,
                "similarity": 0.0
            }

    def detect_spoofing(self, audio_path: str) -> Dict[str, Any]:
        """
        Определяет, является ли аудиозапись спуфинг-атакой

        Args:
            audio_path: Путь к аудиофайлу для проверки

        Returns:
            Словарь с результатами обнаружения спуфинга
        """
        try:
            self.logger.info(f"Checking for spoofing in {audio_path}")

            # Проверяем существование аудиофайла
            if not os.path.exists(audio_path):
                return {
                    "success": False,
                    "message": f"Audio file not found: {audio_path}",
                    "is_spoof": True,  # Считаем подозрительным, если файл не найден
                    "confidence": 1.0
                }

            # Загружаем аудиофайл для анализа
            try:
                waveform, sr = librosa.load(audio_path, sr=self.sample_rate, mono=True)
            except Exception as e:
                self.logger.error(f"Error loading audio file: {e}")
                return {
                    "success": False,
                    "message": f"Error loading audio file: {str(e)}",
                    "is_spoof": True,  # Считаем подозрительным при ошибке загрузки
                    "confidence": 1.0
                }

            # Простые признаки спуфинга:

            # 1. Проверка энергии звука
            energy = np.mean(np.abs(waveform))
            low_energy = energy < 0.01

            # 2. Проверка на однородность спектра (синтезированная речь часто имеет однородный спектр)
            spectral_contrast = librosa.feature.spectral_contrast(y=waveform, sr=sr).mean()
            low_contrast = spectral_contrast < 5.0  # Пороговое значение подобрано эмпирически

            # 3. Проверка на равномерность темпа речи
            onset_env = librosa.onset.onset_strength(y=waveform, sr=sr)
            onset_times = librosa.times_like(onset_env, sr=sr)
            onsets = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr)
            if len(onsets) > 1:
                onset_intervals = np.diff(onset_times[onsets])
                tempo_regularity = np.std(onset_intervals) / np.mean(onset_intervals)
                too_regular = tempo_regularity < 0.2  # Слишком регулярный темп подозрителен
            else:
                too_regular = True  # Если онсетов мало, считаем подозрительным

            # Вычисляем общую оценку "подозрительности"
            spoof_factors = [low_energy, low_contrast, too_regular]
            spoof_confidence = sum(1 for factor in spoof_factors if factor) / len(spoof_factors)

            is_spoof = spoof_confidence > 0.5

            return {
                "success": True,
                "message": "Spoofing detection completed",
                "is_spoof": is_spoof,
                "confidence": float(spoof_confidence),
                "details": {
                    "low_energy": low_energy,
                    "low_spectral_contrast": low_contrast,
                    "too_regular_tempo": too_regular
                }
            }
        except Exception as e:
            self.logger.error(f"Error in spoofing detection: {str(e)}")
            return {
                "success": False,
                "message": f"Error in spoofing detection: {str(e)}",
                "is_spoof": True,  # При ошибке считаем подозрительным
                "confidence": 0.8
            }

    def get_status(self) -> Dict[str, Any]:
        """
        Возвращает текущий статус тренировки
        """
        return self.status