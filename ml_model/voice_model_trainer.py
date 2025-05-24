# ml_model/voice_model_trainer.py
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import logging
import time
from pathlib import Path
import json
import shutil
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
from tqdm import tqdm

# Импорт необходимых библиотек для обработки аудио
import librosa
import soundfile as sf
from voice_embedding import VoiceEmbeddingModel  # Импорт модели для извлечения эмбеддингов

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("voice_model_trainer")


class VoiceDataset(Dataset):
    """
    Датасет для обучения модели распознавания голоса
    """

    def __init__(self, audio_files_by_user, sample_rate=16000, duration=3.0):
        self.audio_files_by_user = audio_files_by_user
        self.sample_rate = sample_rate
        self.duration = duration
        self.target_len = int(duration * sample_rate)

        # Создаем список всех аудиофайлов с метками пользователей
        self.samples = []
        self.user_to_idx = {}

        user_idx = 0
        for user_id, files in audio_files_by_user.items():
            if not files:  # Пропускаем пользователей без файлов
                continue

            self.user_to_idx[user_id] = user_idx

            for file_path in files:
                self.samples.append((file_path, user_idx))

            user_idx += 1

        self.num_users = len(self.user_to_idx)
        logger.info(f"Dataset created with {len(self.samples)} samples from {self.num_users} users")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        file_path, user_idx = self.samples[idx]

        # Загрузка аудио
        try:
            waveform, sr = librosa.load(file_path, sr=self.sample_rate, mono=True)

            # Регулируем длительность
            if len(waveform) > self.target_len:
                # Случайное обрезание для аугментации
                start = np.random.randint(0, len(waveform) - self.target_len)
                waveform = waveform[start:start + self.target_len]
            else:
                # Дополняем нулями до нужной длины
                waveform = np.pad(waveform, (0, max(0, self.target_len - len(waveform))))

            # Нормализация
            waveform = librosa.util.normalize(waveform)

            return torch.FloatTensor(waveform), torch.tensor(user_idx, dtype=torch.long)

        except Exception as e:
            logger.error(f"Error loading file {file_path}: {e}")
            # Возвращаем случайный шум при ошибке
            waveform = np.random.randn(self.target_len) * 0.01
            return torch.FloatTensor(waveform), torch.tensor(user_idx, dtype=torch.long)


class VoiceClassificationModel(nn.Module):
    """
    Простая модель для классификации голосов пользователей
    """

    def __init__(self, num_users, embedding_dim=192):
        super(VoiceClassificationModel, self).__init__()

        self.embedding_dim = embedding_dim
        self.num_users = num_users

        # CNN слои для извлечения признаков
        self.feature_extractor = nn.Sequential(
            # Первый блок
            nn.Conv1d(1, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(3, stride=2, padding=1),

            # Второй блок
            nn.Conv1d(64, 128, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(3, stride=2, padding=1),

            # Третий блок
            nn.Conv1d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.MaxPool1d(3, stride=2, padding=1),

            # Четвертый блок
            nn.Conv1d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )

        # Слои для создания эмбеддингов
        self.embedding_layer = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, embedding_dim),
            nn.ReLU()
        )

        # Классификационный слой
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(embedding_dim, num_users)
        )

        self._initialize_weights()

    def _initialize_weights(self):
        """Инициализация весов"""
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """
        Forward pass
        Args:
            x: Input waveform [batch_size, seq_len]
        Returns:
            embeddings: Voice embeddings [batch_size, embedding_dim]
            logits: Classification logits [batch_size, num_users]
        """
        # Добавляем канальное измерение
        x = x.unsqueeze(1)  # [batch_size, 1, seq_len]

        # Извлечение признаков
        features = self.feature_extractor(x)  # [batch_size, 512, 1]
        features = features.squeeze(-1)  # [batch_size, 512]

        # Создание эмбеддингов
        embeddings = self.embedding_layer(features)  # [batch_size, embedding_dim]

        # Классификация
        logits = self.classifier(embeddings)  # [batch_size, num_users]

        return embeddings, logits


class VoiceModelTrainer:
    """
    Класс для тренировки модели распознавания голоса
    """

    def __init__(
            self,
            model_path: str,
            audio_path: str,
            output_path: str,
            device: Optional[torch.device] = None,
            batch_size: int = 32,
            learning_rate: float = 0.001,
            num_epochs: int = 50
    ):
        self.model_path = model_path
        self.audio_path = audio_path
        self.output_path = output_path
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs

        # Параметры обработки аудио
        self.sample_rate = 16000

        # Статус задачи
        self.status = {
            "status": "initialized",
            "progress": 0.0,
            "message": "Trainer initialized",
            "start_time": None,
            "end_time": None,
            "training_loss": [],
            "validation_loss": [],
            "training_accuracy": [],
            "validation_accuracy": [],
            "best_loss": None
        }

        logger.info(f"Voice model trainer initialized (device: {self.device})")

    def _load_or_create_model(self, num_users: int) -> VoiceClassificationModel:
        """
        Загружает существующую модель или создает новую
        """
        try:
            model = VoiceClassificationModel(num_users=num_users).to(self.device)

            existing_model_path = os.path.join(self.model_path, "voice_classification_model.pt")
            if os.path.exists(existing_model_path):
                try:
                    logger.info("Loading existing voice classification model...")
                    state_dict = torch.load(existing_model_path, map_location=self.device)

                    # Проверяем совместимость размеров
                    if 'classifier.1.weight' in state_dict:
                        existing_num_users = state_dict['classifier.1.weight'].shape[0]
                        if existing_num_users != num_users:
                            logger.warning(
                                f"Model was trained for {existing_num_users} users, but we have {num_users} users. Creating new model.")
                            return model

                    model.load_state_dict(state_dict)
                    logger.info("Existing voice classification model loaded successfully")
                except Exception as e:
                    logger.warning(f"Could not load existing model: {e}. Creating new model.")
            else:
                logger.info("Using new voice classification model")

            return model
        except Exception as e:
            logger.error(f"Error loading or creating voice classification model: {e}")
            raise

    def _prepare_data(self, user_ids: List[str]) -> Tuple[DataLoader, DataLoader]:
        """
        Подготовка данных для тренировки модели распознавания голоса
        """
        self.status["status"] = "preparing_data"
        self.status["message"] = "Preparing training data"

        try:
            # Сбор аудиофайлов для каждого пользователя
            audio_files_by_user = {}

            for user_id in user_ids:
                user_dir = os.path.join(self.audio_path, user_id)

                if not os.path.exists(user_dir):
                    logger.warning(f"User directory not found: {user_dir}")
                    continue

                # Поиск аудиофайлов
                audio_files = []
                for filename in os.listdir(user_dir):
                    if filename.lower().endswith((".wav", ".mp3", ".flac")):
                        file_path = os.path.join(user_dir, filename)
                        if os.path.getsize(file_path) > 1000:  # Файл не пустой
                            audio_files.append(file_path)

                if len(audio_files) >= 3:  # Минимум 3 файла на пользователя
                    audio_files_by_user[user_id] = audio_files
                    logger.info(f"User {user_id}: {len(audio_files)} audio files")
                else:
                    logger.warning(f"User {user_id} has insufficient audio files ({len(audio_files)})")

            if len(audio_files_by_user) < 2:
                raise ValueError(
                    f"Insufficient users with enough audio files. Need at least 2, got {len(audio_files_by_user)}")

            # Создание датасета
            dataset = VoiceDataset(audio_files_by_user, self.sample_rate)

            # Разделение на обучающую и валидационную выборки
            dataset_size = len(dataset)
            train_size = int(0.8 * dataset_size)
            val_size = dataset_size - train_size

            train_dataset, val_dataset = torch.utils.data.random_split(
                dataset, [train_size, val_size]
            )

            # Создание загрузчиков данных
            train_loader = DataLoader(
                train_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=2 if os.name != 'nt' else 0,
                pin_memory=True if self.device.type == 'cuda' else False
            )

            val_loader = DataLoader(
                val_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=2 if os.name != 'nt' else 0,
                pin_memory=True if self.device.type == 'cuda' else False
            )

            logger.info(f"Prepared data loaders: {len(train_dataset)} training, {len(val_dataset)} validation samples")

            return train_loader, val_loader, dataset.num_users, dataset.user_to_idx

        except Exception as e:
            self.status["status"] = "error"
            self.status["message"] = f"Error preparing data: {str(e)}"
            logger.error(f"Error preparing data: {e}")
            raise

    # ml_model/voice_model_trainer.py - Добавление поддержки callback для прогресса

    def train(self, user_ids: List[str], progress_callback=None) -> Dict[str, Any]:
        """
        Обучает модель распознавания голоса с поддержкой callback для прогресса
        """
        self.status["start_time"] = datetime.now().isoformat()
        self.status["status"] = "training"
        self.status["message"] = "Starting voice model training"

        try:
            # Подготовка данных
            train_loader, val_loader, num_users, user_to_idx = self._prepare_data(user_ids)

            # Загрузка или создание модели
            model = self._load_or_create_model(num_users)
            model.train()

            # Оптимизатор и функция потерь
            optimizer = optim.Adam(model.parameters(), lr=self.learning_rate, weight_decay=1e-5)
            criterion = nn.CrossEntropyLoss()
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

            # Обучение модели
            best_val_loss = float('inf')
            best_val_acc = 0.0
            patience = 10
            patience_counter = 0

            for epoch in range(self.num_epochs):
                # Тренировочный цикл
                model.train()
                train_loss = 0.0
                train_correct = 0
                train_total = 0

                for batch_idx, (waveforms, labels) in enumerate(train_loader):
                    waveforms = waveforms.to(self.device)
                    labels = labels.to(self.device)

                    # Forward pass
                    embeddings, logits = model(waveforms)
                    loss = criterion(logits, labels)

                    # Backward pass и оптимизация
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    # Статистика
                    train_loss += loss.item()
                    _, predicted = torch.max(logits.data, 1)
                    train_total += labels.size(0)
                    train_correct += (predicted == labels).sum().item()

                train_loss /= len(train_loader)
                train_accuracy = 100.0 * train_correct / train_total

                # Валидация
                model.eval()
                val_loss = 0.0
                val_correct = 0
                val_total = 0

                with torch.no_grad():
                    for waveforms, labels in val_loader:
                        waveforms = waveforms.to(self.device)
                        labels = labels.to(self.device)

                        # Forward pass
                        embeddings, logits = model(waveforms)
                        loss = criterion(logits, labels)

                        # Статистика
                        val_loss += loss.item()
                        _, predicted = torch.max(logits.data, 1)
                        val_total += labels.size(0)
                        val_correct += (predicted == labels).sum().item()

                val_loss /= len(val_loader)
                val_accuracy = 100.0 * val_correct / val_total

                # Обновление learning rate
                scheduler.step()

                # ВАЖНО: Вызов callback для обновления прогресса
                if progress_callback:
                    try:
                        progress_callback(epoch + 1, self.num_epochs, train_loss, train_accuracy)
                    except Exception as cb_error:
                        logger.warning(f"Progress callback error: {cb_error}")

                # Сохранение лучшей модели
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_val_acc = val_accuracy
                    patience_counter = 0

                    # Сохранение модели
                    os.makedirs(self.output_path, exist_ok=True)
                    torch.save(model.state_dict(), os.path.join(self.output_path, "voice_classification_model.pt"))

                    # Сохранение маппинга пользователей
                    with open(os.path.join(self.output_path, "user_mapping.json"), "w") as f:
                        json.dump(user_to_idx, f, indent=4)

                    logger.info(f"Saved best voice model (val_loss: {val_loss:.4f}, val_accuracy: {val_accuracy:.2f}%)")
                else:
                    patience_counter += 1

                # Обновление внутреннего статуса
                self.status["progress"] = (epoch + 1) / self.num_epochs * 100
                self.status[
                    "message"] = f"Epoch {epoch + 1}/{self.num_epochs}, train_acc: {train_accuracy:.2f}%, val_acc: {val_accuracy:.2f}%"
                self.status["training_loss"].append(train_loss)
                self.status["validation_loss"].append(val_loss)
                self.status["training_accuracy"].append(train_accuracy)
                self.status["validation_accuracy"].append(val_accuracy)
                self.status["best_loss"] = best_val_loss

                logger.info(
                    f"Epoch {epoch + 1}/{self.num_epochs}, train_loss: {train_loss:.4f}, val_loss: {val_loss:.4f}, "
                    f"train_acc: {train_accuracy:.2f}%, val_acc: {val_accuracy:.2f}%")

                # Early stopping
                if patience_counter >= patience:
                    logger.info(f"Early stopping after {epoch + 1} epochs")
                    # Финальный вызов callback
                    if progress_callback:
                        try:
                            progress_callback(self.num_epochs, self.num_epochs, best_val_loss, best_val_acc)
                        except Exception as cb_error:
                            logger.warning(f"Final progress callback error: {cb_error}")
                    break

            # Копирование лучшей модели в основную директорию
            os.makedirs(self.model_path, exist_ok=True)
            shutil.copy(
                os.path.join(self.output_path, "voice_classification_model.pt"),
                os.path.join(self.model_path, "voice_classification_model.pt")
            )
            shutil.copy(
                os.path.join(self.output_path, "user_mapping.json"),
                os.path.join(self.model_path, "user_mapping.json")
            )

            # Создание конфигурационного файла
            config = {
                "model_type": "VoiceClassificationModel",
                "num_users": num_users,
                "embedding_dim": 192,
                "sample_rate": self.sample_rate,
                "trained_date": datetime.now().isoformat(),
                "best_val_loss": best_val_loss,
                "best_val_accuracy": best_val_acc,
                "user_mapping": user_to_idx
            }

            with open(os.path.join(self.model_path, "voice_model_config.json"), "w") as f:
                json.dump(config, f, indent=4)

            # Обновление статуса
            self.status["status"] = "completed"
            self.status["progress"] = 100.0  # Убедимся, что прогресс 100%
            self.status["message"] = f"Voice model training completed successfully. Best accuracy: {best_val_acc:.2f}%"
            self.status["end_time"] = datetime.now().isoformat()

            logger.info("Voice model training completed successfully")

            return self.status

        except Exception as e:
            self.status["status"] = "error"
            self.status["progress"] = 0.0
            self.status["message"] = f"Error during voice model training: {str(e)}"
            self.status["end_time"] = datetime.now().isoformat()
            logger.error(f"Error during voice model training: {e}")
            raise

    def get_status(self) -> Dict[str, Any]:
        """
        Возвращает текущий статус тренировки
        """
        return self.status