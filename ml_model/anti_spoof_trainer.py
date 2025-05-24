# ml_model/anti_spoof_trainer.py - Исправленная версия

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
from anti_spoof import ImprovedAntiSpoofingNet, RawNet2AntiSpoofing  # Импорт модели для защиты от спуфинга

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("anti_spoof_trainer")


class AntiSpoofDataset(Dataset):
    """
    Датасет для обучения модели защиты от спуфинга
    """

    def __init__(self, real_files, spoof_files, sample_rate=16000, duration=3.0):
        self.real_files = real_files
        self.spoof_files = spoof_files
        self.sample_rate = sample_rate
        self.duration = duration
        self.target_len = int(duration * sample_rate)

        logger.info(f"Dataset created with {len(real_files)} real and {len(spoof_files)} spoof files")

    def __len__(self):
        return len(self.real_files) + len(self.spoof_files)

    def __getitem__(self, idx):
        # Определяем, это реальный или поддельный файл
        is_real = idx < len(self.real_files)

        if is_real:
            file_path = self.real_files[idx]
            label = 0  # 0 = реальный
        else:
            file_path = self.spoof_files[idx - len(self.real_files)]
            label = 1  # 1 = поддельный

        # Загрузка аудио
        try:
            waveform, sr = librosa.load(file_path, sr=self.sample_rate, mono=True)

            # Регулируем длительность
            if len(waveform) > self.target_len:
                # Случайное обрезание для аугментации
                start = np.random.randint(0, len(waveform) - self.target_len)
                waveform = waveform[start:start + self.target_len]
            else:
                # Дополняем нулями до нужной длены
                waveform = np.pad(waveform, (0, max(0, self.target_len - len(waveform))))

            # Нормализация
            waveform = librosa.util.normalize(waveform)

            return torch.FloatTensor(waveform), torch.tensor(label, dtype=torch.float32)
        except Exception as e:
            logger.error(f"Error loading file {file_path}: {e}")
            # Возвращаем случайный шум при ошибке
            waveform = np.random.randn(self.target_len)
            return torch.FloatTensor(waveform), torch.tensor(label, dtype=torch.float32)


class AntiSpoofTrainer:
    """
    Класс для тренировки модели защиты от спуфинга с поддержкой RawNet
    """

    def __init__(
            self,
            model_path: str,
            real_audio_path: str,
            spoof_audio_path: str,
            output_path: str,
            device: Optional[torch.device] = None,
            batch_size: int = 32,
            learning_rate: float = 0.001,
            num_epochs: int = 50,
            use_pretrained_rawnet: bool = True
    ):
        self.model_path = model_path
        self.real_audio_path = real_audio_path
        self.spoof_audio_path = spoof_audio_path
        self.output_path = output_path
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.use_pretrained_rawnet = use_pretrained_rawnet

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
            "validation_accuracy": []
        }

        logger.info(f"Anti-spoof trainer initialized (device: {self.device})")

    def _load_or_create_model(self) -> nn.Module:
        """
        Загружает существующую модель или создает новую с поддержкой RawNet
        """
        try:
            if self.use_pretrained_rawnet:
                # Пытаемся загрузить предобученную RawNet модель
                model = self._load_pretrained_rawnet()
            else:
                # Используем улучшенную модель
                model = ImprovedAntiSpoofingNet().to(self.device)

            # Проверяем наличие сохраненных весов
            saved_model_path = os.path.join(self.model_path, "anti_spoof_model.pt")
            if os.path.exists(saved_model_path):
                try:
                    logger.info("Loading existing anti-spoof model...")
                    state_dict = torch.load(saved_model_path, map_location=self.device)
                    model.load_state_dict(state_dict, strict=False)  # strict=False для совместимости
                    logger.info("Existing anti-spoof model loaded successfully")
                except Exception as e:
                    logger.warning(f"Could not load existing model: {e}. Using new model.")
            else:
                logger.info("Using new anti-spoof model")

            return model
        except Exception as e:
            logger.error(f"Error loading or creating anti-spoof model: {e}")
            # Возвращаем базовую модель при ошибке
            return ImprovedAntiSpoofingNet().to(self.device)

    def _load_pretrained_rawnet(self):
        """
        Загружает предобученную RawNet модель
        """
        try:
            # Путь к предобученной модели RawNet
            pretrained_path = os.path.join(self.model_path, "rawnet2_pretrained.pt")

            if os.path.exists(pretrained_path):
                logger.info(f"Loading pretrained RawNet from {pretrained_path}")

                # Создаем модель RawNet2
                model = RawNet2AntiSpoofing().to(self.device)

                # Загружаем предобученные веса
                checkpoint = torch.load(pretrained_path, map_location=self.device)

                # Обрабатываем различные форматы checkpoint
                if 'model_state_dict' in checkpoint:
                    state_dict = checkpoint['model_state_dict']
                elif 'state_dict' in checkpoint:
                    state_dict = checkpoint['state_dict']
                else:
                    state_dict = checkpoint

                # Загружаем веса с обработкой несовместимых ключей
                model.load_state_dict(state_dict, strict=False)

                logger.info("Pretrained RawNet loaded successfully")
                return model
            else:
                logger.warning(f"Pretrained RawNet not found at {pretrained_path}")
                logger.info("Downloading pretrained RawNet weights...")

                # Создаем модель и возвращаем ее (можно добавить автозагрузку)
                model = RawNet2AntiSpoofing().to(self.device)

                # TODO: Добавить автозагрузку предобученных весов RawNet2
                # Пока используем случайную инициализацию
                logger.warning("Using randomly initialized RawNet model")

                return model

        except Exception as e:
            logger.error(f"Error loading pretrained RawNet: {e}")
            # Возвращаем улучшенную модель при ошибке
            return ImprovedAntiSpoofingNet().to(self.device)

    def _prepare_data(self) -> Tuple[DataLoader, DataLoader]:
        """
        Подготовка данных для тренировки модели защиты от спуфинга
        """
        self.status["status"] = "preparing_data"
        self.status["message"] = "Preparing training data"

        try:
            # Проверка путей
            if not os.path.exists(self.real_audio_path):
                raise FileNotFoundError(f"Real audio path not found: {self.real_audio_path}")

            if not os.path.exists(self.spoof_audio_path):
                # Если путь к поддельным аудио не существует, создаем его
                os.makedirs(self.spoof_audio_path, exist_ok=True)
                logger.warning(f"Spoof audio path not found, created new directory: {self.spoof_audio_path}")

            # Поиск реальных аудиофайлов
            real_files = []
            for root, _, files in os.walk(self.real_audio_path):
                for file in files:
                    if file.endswith((".wav", ".WAV")):
                        real_files.append(os.path.join(root, file))

            # Поиск поддельных аудиофайлов
            spoof_files = []
            for root, _, files in os.walk(self.spoof_audio_path):
                for file in files:
                    if file.endswith((".wav", ".WAV")):
                        spoof_files.append(os.path.join(root, file))

            logger.info(f"Found {len(real_files)} real audio files and {len(spoof_files)} spoof audio files")

            # Если не хватает поддельных аудио, генерируем синтетические
            if len(spoof_files) < len(real_files) * 0.2:  # Минимум 20% от реальных
                needed_spoof = int(len(real_files) * 0.5) - len(spoof_files)
                logger.warning(f"Not enough spoof files. Need to generate {needed_spoof} synthetic spoof samples")

                # Создаем синтетические спуфинг данные из реальных аудио
                synthetic_spoof_files = self._generate_synthetic_spoof(real_files, needed_spoof)
                spoof_files.extend(synthetic_spoof_files)
                logger.info(f"Generated {len(synthetic_spoof_files)} synthetic spoof files")

            # Если все равно мало спуфинг данных, берем часть реальных данных
            if len(spoof_files) < len(real_files) * 0.2:
                logger.warning("Still not enough spoof files, using a subset of real files")
                real_files = real_files[:len(spoof_files) * 5]  # 5:1 соотношение

            # Разделение на обучающую и валидационную выборки
            train_real = real_files[:int(len(real_files) * 0.8)]
            val_real = real_files[int(len(real_files) * 0.8):]

            train_spoof = spoof_files[:int(len(spoof_files) * 0.8)]
            val_spoof = spoof_files[int(len(spoof_files) * 0.8):]

            # Создание датасетов
            train_dataset = AntiSpoofDataset(train_real, train_spoof, self.sample_rate)
            val_dataset = AntiSpoofDataset(val_real, val_spoof, self.sample_rate)

            # Создание загрузчиков данных
            train_loader = DataLoader(
                train_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=2 if os.name != 'nt' else 0  # На Windows нужно 0
            )

            val_loader = DataLoader(
                val_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=2 if os.name != 'nt' else 0
            )

            logger.info(f"Prepared data loaders: {len(train_dataset)} training, {len(val_dataset)} validation samples")

            return train_loader, val_loader

        except Exception as e:
            self.status["status"] = "error"
            self.status["message"] = f"Error preparing data: {str(e)}"
            logger.error(f"Error preparing data: {e}")
            raise

    def _generate_synthetic_spoof(self, real_files: List[str], count: int) -> List[str]:
        """
        Генерирует синтетические спуфинг аудио из реальных файлов
        """
        synthetic_files = []

        # Создаем каталог для синтетических спуфинг данных, если не существует
        synth_dir = os.path.join(self.spoof_audio_path, "synthetic")
        os.makedirs(synth_dir, exist_ok=True)

        # Выбираем случайные файлы из реальных
        selected_files = np.random.choice(real_files, min(count, len(real_files)), replace=count > len(real_files))

        for i, file_path in enumerate(selected_files):
            try:
                # Загрузка аудио
                waveform, sr = librosa.load(file_path, sr=self.sample_rate, mono=True)

                # Синтетическая атака (простая реализация)
                attack_type = np.random.choice(["playback", "pitch_shift", "time_stretch", "noise"])

                if attack_type == "playback":
                    # Имитация записи через динамик
                    reverb = np.random.uniform(0.1, 0.5)
                    noise_level = np.random.uniform(0.01, 0.05)

                    # Добавление реверберации
                    impulse_response = np.exp(-np.linspace(0, 1, int(sr * reverb)))
                    impulse_response = impulse_response / np.sum(impulse_response)
                    waveform = np.convolve(waveform, impulse_response, mode='full')[:len(waveform)]

                    # Добавление шума
                    waveform = waveform + np.random.normal(0, noise_level, len(waveform))

                elif attack_type == "pitch_shift":
                    # Изменение высоты голоса
                    n_steps = np.random.uniform(-4, 4)
                    waveform = librosa.effects.pitch_shift(waveform, sr=sr, n_steps=n_steps)

                elif attack_type == "time_stretch":
                    # Растяжение/сжатие по времени
                    rate = np.random.uniform(0.8, 1.2)
                    waveform = librosa.effects.time_stretch(waveform, rate=rate)

                elif attack_type == "noise":
                    # Добавление шума
                    noise_level = np.random.uniform(0.05, 0.2)
                    waveform = waveform + np.random.normal(0, noise_level, len(waveform))

                # Нормализация
                waveform = librosa.util.normalize(waveform)

                # Сохранение синтетического спуфинг аудио
                out_file = os.path.join(synth_dir, f"synth_spoof_{i}_{attack_type}.wav")
                sf.write(out_file, waveform, sr)

                synthetic_files.append(out_file)

                if i % 10 == 0:
                    logger.info(f"Generated {i + 1}/{count} synthetic spoof files")

            except Exception as e:
                logger.error(f"Error generating synthetic spoof for {file_path}: {e}")

        return synthetic_files

    def train(self, task_id: str = None, progress_callback=None) -> Dict[str, Any]:
        """
        Обучает модель защиты от спуфинга с поддержкой progress_callback
        """
        self.status["start_time"] = datetime.now().isoformat()
        self.status["status"] = "training"
        self.status["message"] = "Starting anti-spoofing model training"

        try:
            # Загрузка или создание модели
            model = self._load_or_create_model()
            model.train()

            # Подготовка данных
            train_loader, val_loader = self._prepare_data()

            # Оптимизатор и функция потерь
            optimizer = optim.Adam(model.parameters(), lr=self.learning_rate)
            criterion = nn.BCELoss()  # Бинарная кросс-энтропия для задачи классификации

            # Обучение модели
            best_val_loss = float('inf')
            patience = 5
            patience_counter = 0

            for epoch in range(self.num_epochs):
                # Тренировочный цикл
                model.train()
                train_loss = 0.0
                train_correct = 0
                train_total = 0

                for waveforms, labels in train_loader:
                    # Перенос данных на устройство
                    waveforms = waveforms.unsqueeze(1).to(self.device)  # [batch, 1, time]
                    labels = labels.to(self.device)

                    # Forward pass
                    outputs = model(waveforms).squeeze()
                    loss = criterion(outputs, labels)

                    # Backward pass и оптимизация
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    # Статистика
                    train_loss += loss.item() * waveforms.size(0)
                    predicted = (outputs >= 0.5).float()
                    train_correct += (predicted == labels).sum().item()
                    train_total += labels.size(0)

                train_loss /= len(train_loader.dataset)
                train_accuracy = train_correct / train_total * 100

                # Валидация
                model.eval()
                val_loss = 0.0
                val_correct = 0
                val_total = 0

                with torch.no_grad():
                    for waveforms, labels in val_loader:
                        # Перенос данных на устройство
                        waveforms = waveforms.unsqueeze(1).to(self.device)
                        labels = labels.to(self.device)

                        # Forward pass
                        outputs = model(waveforms).squeeze()
                        loss = criterion(outputs, labels)

                        # Статистика
                        val_loss += loss.item() * waveforms.size(0)
                        predicted = (outputs >= 0.5).float()
                        val_correct += (predicted == labels).sum().item()
                        val_total += labels.size(0)

                val_loss /= len(val_loader.dataset)
                val_accuracy = val_correct / val_total * 100

                # Вызов callback для обновления прогресса
                if progress_callback:
                    try:
                        progress_callback(epoch + 1, self.num_epochs, train_loss, train_accuracy)
                    except Exception as cb_error:
                        logger.warning(f"Progress callback error: {cb_error}")

                # Сохранение лучшей модели
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0

                    # Сохранение модели
                    os.makedirs(self.output_path, exist_ok=True)
                    torch.save(model.state_dict(), os.path.join(self.output_path, "anti_spoof_model.pt"))
                    logger.info(
                        f"Saved best anti-spoof model (val_loss: {val_loss:.4f}, val_accuracy: {val_accuracy:.2f}%)")
                else:
                    patience_counter += 1

                # Обновление статуса
                self.status["progress"] = (epoch + 1) / self.num_epochs * 100
                self.status[
                    "message"] = f"Epoch {epoch + 1}/{self.num_epochs}, train_acc: {train_accuracy:.2f}%, val_acc: {val_accuracy:.2f}%"
                self.status["training_loss"].append(train_loss)
                self.status["validation_loss"].append(val_loss)
                self.status["training_accuracy"].append(train_accuracy)
                self.status["validation_accuracy"].append(val_accuracy)

                logger.info(
                    f"Epoch {epoch + 1}/{self.num_epochs}, train_loss: {train_loss:.4f}, val_loss: {val_loss:.4f}, "
                    f"train_acc: {train_accuracy:.2f}%, val_acc: {val_accuracy:.2f}%")

                # Early stopping
                if patience_counter >= patience:
                    logger.info(f"Early stopping after {epoch + 1} epochs")
                    break

            # Копирование лучшей модели в основную директорию
            shutil.copy(
                os.path.join(self.output_path, "anti_spoof_model.pt"),
                os.path.join(self.model_path, "anti_spoof_model.pt")
            )

            # Создание конфигурационного файла
            config = {
                "model_type": "RawNet2AntiSpoofing" if self.use_pretrained_rawnet else "ImprovedAntiSpoofingNet",
                "sample_rate": self.sample_rate,
                "trained_date": datetime.now().isoformat(),
                "best_val_loss": best_val_loss,
                "best_val_accuracy": val_accuracy,
                "use_pretrained_rawnet": self.use_pretrained_rawnet
            }

            with open(os.path.join(self.model_path, "anti_spoof_config.json"), "w") as f:
                json.dump(config, f, indent=4)

            # Обновление статуса
            self.status["status"] = "completed"
            self.status["message"] = "Anti-spoofing model training completed successfully"
            self.status["progress"] = 100.0
            self.status["end_time"] = datetime.now().isoformat()

            logger.info("Anti-spoofing model training completed successfully")

            return self.status

        except Exception as e:
            self.status["status"] = "error"
            self.status["message"] = f"Error during anti-spoofing training: {str(e)}"
            self.status["end_time"] = datetime.now().isoformat()
            logger.error(f"Error during anti-spoofing training: {e}")
            raise

    def get_status(self) -> Dict[str, Any]:
        """
        Возвращает текущий статус тренировки
        """
        return self.status