# ml_model/anti_spoof_trainer.py - Исправленная версия по аналогии с voice_embedding.py

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
from anti_spoof import ImprovedAntiSpoofingNet, RawNet2AntiSpoofing

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("anti_spoof_trainer")


class AntiSpoofDataset(Dataset):
    """
    Датасет для обучения модели защиты от спуфинга с правильной обработкой данных
    """

    def __init__(self, real_files, spoof_files, sample_rate=16000, duration=3.0, use_rawnet=False):
        self.real_files = real_files
        self.spoof_files = spoof_files
        self.sample_rate = sample_rate
        self.duration = duration
        self.target_len = int(duration * sample_rate)
        self.use_rawnet = use_rawnet

        # Параметры для мел-спектрограммы
        self.n_fft = 512
        self.hop_length = 256
        self.n_mels = 40

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

        # Загрузка аудио - используем тот же подход что и в voice_embedding.py
        try:
            # Загрузка и нормализация аудио с помощью librosa
            waveform, sr = librosa.load(file_path, sr=self.sample_rate, mono=True)

            # Проверка на тишину или слишком короткий файл
            if len(waveform) < 0.5 * self.sample_rate or np.max(np.abs(waveform)) < 0.01:
                logger.warning(f"Audio file {file_path} is too short or contains silence")
                # Генерируем синтетический сигнал для тестирования
                waveform = np.sin(np.linspace(0, 100 * np.pi, self.target_len)) * 0.1

            # Регулируем длительность
            if len(waveform) > self.target_len:
                # Случайное обрезание для аугментации
                start = np.random.randint(0, len(waveform) - self.target_len)
                waveform = waveform[start:start + self.target_len]
            else:
                # Дополняем нулями до нужной длены
                waveform = np.pad(waveform, (0, max(0, self.target_len - len(waveform))))

            # Нормализация амплитуды
            waveform = waveform / (np.max(np.abs(waveform)) + 1e-10)

            # Возвращаем данные в зависимости от типа модели
            if self.use_rawnet:
                # Для RawNet2 возвращаем сырой сигнал
                return torch.FloatTensor(waveform), torch.tensor(label, dtype=torch.float32)
            else:
                # Для CNN+LSTM модели создаем мел-спектрограмму
                mel_spec = librosa.feature.melspectrogram(
                    y=waveform,
                    sr=self.sample_rate,
                    n_fft=self.n_fft,
                    hop_length=self.hop_length,
                    n_mels=self.n_mels,
                    fmax=self.sample_rate // 2
                )

                # Преобразование в логарифмическую шкалу
                log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)

                # Нормализация
                log_mel_spec = (log_mel_spec - np.mean(log_mel_spec)) / (np.std(log_mel_spec) + 1e-8)

                # Приведение к стандартному размеру
                target_time_length = 128
                current_length = log_mel_spec.shape[1]

                if current_length > target_time_length:
                    start_idx = (current_length - target_time_length) // 2
                    log_mel_spec = log_mel_spec[:, start_idx:start_idx + target_time_length]
                elif current_length < target_time_length:
                    pad_width = target_time_length - current_length
                    pad_left = pad_width // 2
                    pad_right = pad_width - pad_left
                    log_mel_spec = np.pad(log_mel_spec, ((0, 0), (pad_left, pad_right)), mode='constant')

                return torch.FloatTensor(log_mel_spec), torch.tensor(label, dtype=torch.float32)

        except Exception as e:
            logger.error(f"Error loading file {file_path}: {e}")
            # Возвращаем случайные данные при ошибке
            if self.use_rawnet:
                waveform = np.random.randn(self.target_len) * 0.1
                return torch.FloatTensor(waveform), torch.tensor(label, dtype=torch.float32)
            else:
                mel_spec = np.random.randn(self.n_mels, 128) * 0.1
                return torch.FloatTensor(mel_spec), torch.tensor(label, dtype=torch.float32)


class AntiSpoofTrainer:
    """
    Класс для тренировки модели защиты от спуфинга по аналогии с voice_embedding.py
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

        # Инициализируем устройство используя безопасный метод
        self.device = self._initialize_device(device)

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

        # Создаем директории
        os.makedirs(self.model_path, exist_ok=True)
        os.makedirs(self.output_path, exist_ok=True)

        logger.info(f"Anti-spoof trainer initialized (device: {self.device})")

    def _initialize_device(self, requested_device):
        """Безопасно инициализирует устройство для вычислений с проверкой CUDA"""
        if requested_device:
            return torch.device(requested_device)

        # Пытаемся использовать CUDA, если доступна
        if torch.cuda.is_available():
            try:
                device = torch.device("cuda")
                test_tensor = torch.zeros(1, 1).to(device)
                _ = test_tensor + 1  # Проверяем, что операции работают
                logger.info("CUDA is available and working properly")
                return device
            except Exception as e:
                logger.warning(f"CUDA error: {e}, falling back to CPU")

        # Используем CPU, если CUDA недоступна или с ней проблемы
        logger.info("Using CPU for computations")
        return torch.device("cpu")

    def _load_or_create_model(self) -> nn.Module:
        """
        Загружает существующую модель или создает новую с поддержкой разных архитектур
        """
        try:
            # Выбираем тип модели
            if self.use_pretrained_rawnet:
                logger.info("Trying to initialize RawNet2 model")
                try:
                    model = RawNet2AntiSpoofing().to(self.device)
                    self.using_rawnet = True
                    logger.info("RawNet2 model initialized successfully")
                except Exception as rawnet_error:
                    logger.warning(f"Could not initialize RawNet2: {rawnet_error}")
                    logger.info("Falling back to CNN+LSTM model")
                    model = ImprovedAntiSpoofingNet().to(self.device)
                    self.using_rawnet = False
            else:
                # Используем улучшенную CNN+LSTM модель
                model = ImprovedAntiSpoofingNet().to(self.device)
                self.using_rawnet = False
                logger.info("Using CNN+LSTM model")

            # Проверяем наличие сохраненных весов
            if self.using_rawnet:
                saved_model_path = os.path.join(self.model_path, "rawnet2_antispoof.pt")
            else:
                saved_model_path = os.path.join(self.model_path, "anti_spoof_model.pt")

            if os.path.exists(saved_model_path):
                try:
                    logger.info(f"Loading existing model from {saved_model_path}")
                    state_dict = torch.load(saved_model_path, map_location=self.device)
                    model.load_state_dict(state_dict, strict=False)  # strict=False для совместимости
                    logger.info("Existing model loaded successfully")
                except Exception as e:
                    logger.warning(f"Could not load existing model: {e}. Using new model.")
            else:
                logger.info("Using new model")

            return model
        except Exception as e:
            logger.error(f"Error loading or creating model: {e}")
            # Возвращаем базовую модель при ошибке
            self.using_rawnet = False
            return ImprovedAntiSpoofingNet().to(self.device)

    def _prepare_data(self) -> Tuple[DataLoader, DataLoader]:
        """
        Подготовка данных для тренировки с улучшенной обработкой
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
                    if file.endswith((".wav", ".WAV", ".mp3", ".flac")):
                        real_files.append(os.path.join(root, file))

            # Поиск поддельных аудиофайлов
            spoof_files = []
            for root, _, files in os.walk(self.spoof_audio_path):
                for file in files:
                    if file.endswith((".wav", ".WAV", ".mp3", ".flac")):
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

            # Создание датасетов с правильным флагом для типа модели
            train_dataset = AntiSpoofDataset(
                train_real, train_spoof,
                self.sample_rate,
                use_rawnet=hasattr(self, 'using_rawnet') and self.using_rawnet
            )
            val_dataset = AntiSpoofDataset(
                val_real, val_spoof,
                self.sample_rate,
                use_rawnet=hasattr(self, 'using_rawnet') and self.using_rawnet
            )

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

                # Проверяем на тишину
                if len(waveform) < 0.5 * self.sample_rate or np.max(np.abs(waveform)) < 0.01:
                    # Создаем синтетический сигнал
                    waveform = np.sin(np.linspace(0, 100 * np.pi, self.sample_rate * 3)) * 0.1

                # Синтетическая атака (улучшенная реализация)
                attack_type = np.random.choice(["playback", "pitch_shift", "time_stretch", "noise", "vocoder"])

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

                elif attack_type == "vocoder":
                    # Имитация вокодера
                    # Простая имитация путем квантования амплитуды
                    quantization_levels = np.random.randint(8, 32)
                    waveform = np.round(waveform * quantization_levels) / quantization_levels

                # Нормализация
                waveform = waveform / (np.max(np.abs(waveform)) + 1e-10)

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
        Обучает модель защиты от спуфинга с улучшенной обработкой ошибок
        """
        self.status["start_time"] = datetime.now().isoformat()
        self.status["status"] = "training"
        self.status["message"] = "Starting anti-spoofing model training"

        try:
            # Загрузка или создание модели
            model = self._load_or_create_model()
            model.train()

            # Определяем тип модели для правильной обработки данных
            is_rawnet = hasattr(self, 'using_rawnet') and self.using_rawnet
            logger.info(f"Using model type: {'RawNet2' if is_rawnet else 'CNN+LSTM'}")

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
                train_batches = 0

                for batch_idx, (data, labels) in enumerate(train_loader):
                    try:
                        # Правильная подготовка данных в зависимости от типа модели
                        if is_rawnet:
                            # Для RawNet2: [batch_size, samples]
                            if data.dim() == 2:  # [batch_size, samples]
                                waveforms = data.to(self.device)
                            else:  # убираем лишние измерения
                                waveforms = data.squeeze().to(self.device)
                                if waveforms.dim() == 1:  # Если остался 1D, добавляем batch dimension
                                    waveforms = waveforms.unsqueeze(0)
                        else:
                            # Для CNN+LSTM: [batch_size, 1, time, freq]
                            if data.dim() == 3:  # [batch_size, time, freq]
                                waveforms = data.unsqueeze(1).to(self.device)
                            else:  # [batch_size, time, freq] или другое
                                waveforms = data.to(self.device)
                                if waveforms.dim() == 3:
                                    waveforms = waveforms.unsqueeze(1)

                        labels = labels.to(self.device)

                        # Forward pass
                        outputs = model(waveforms)
                        if outputs.dim() > 1:
                            outputs = outputs.squeeze()

                        # Убеждаемся что размерности совпадают
                        if outputs.shape != labels.shape:
                            if outputs.dim() == 0:  # скалярный тензор
                                outputs = outputs.unsqueeze(0)
                            if labels.dim() == 0:
                                labels = labels.unsqueeze(0)

                        loss = criterion(outputs, labels)

                        # Backward pass и оптимизация
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

                        # Статистика
                        train_loss += loss.item()
                        predicted = (outputs >= 0.5).float()
                        train_correct += (predicted == labels).sum().item()
                        train_total += labels.size(0)
                        train_batches += 1

                    except Exception as batch_error:
                        logger.error(f"Error in training batch {batch_idx}: {batch_error}")
                        logger.error(f"Data shape: {data.shape}, Labels shape: {labels.shape}")
                        continue

                if train_batches > 0:
                    train_loss /= train_batches
                    train_accuracy = train_correct / train_total * 100 if train_total > 0 else 0
                else:
                    logger.error("No successful training batches!")
                    break

                # Валидация
                model.eval()
                val_loss = 0.0
                val_correct = 0
                val_total = 0
                val_batches = 0

                with torch.no_grad():
                    for data, labels in val_loader:
                        try:
                            # Правильная подготовка данных для валидации
                            if is_rawnet:
                                if data.dim() == 2:
                                    waveforms = data.to(self.device)
                                else:
                                    waveforms = data.squeeze().to(self.device)
                                    if waveforms.dim() == 1:
                                        waveforms = waveforms.unsqueeze(0)
                            else:
                                if data.dim() == 3:
                                    waveforms = data.unsqueeze(1).to(self.device)
                                else:
                                    waveforms = data.to(self.device)
                                    if waveforms.dim() == 3:
                                        waveforms = waveforms.unsqueeze(1)

                            labels = labels.to(self.device)

                            # Forward pass
                            outputs = model(waveforms)
                            if outputs.dim() > 1:
                                outputs = outputs.squeeze()

                            # Убеждаемся что размерности совпадают
                            if outputs.shape != labels.shape:
                                if outputs.dim() == 0:
                                    outputs = outputs.unsqueeze(0)
                                if labels.dim() == 0:
                                    labels = labels.unsqueeze(0)

                            loss = criterion(outputs, labels)

                            # Статистика
                            val_loss += loss.item()
                            predicted = (outputs >= 0.5).float()
                            val_correct += (predicted == labels).sum().item()
                            val_total += labels.size(0)
                            val_batches += 1

                        except Exception as val_error:
                            logger.error(f"Validation error: {val_error}")
                            continue

                if val_batches > 0:
                    val_loss /= val_batches
                    val_accuracy = val_correct / val_total * 100 if val_total > 0 else 0
                else:
                    logger.warning("No successful validation batches!")
                    val_loss = float('inf')
                    val_accuracy = 0

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
                    try:
                        model_filename = "rawnet2_antispoof.pt" if is_rawnet else "anti_spoof_model.pt"
                        torch.save(model.state_dict(), os.path.join(self.output_path, model_filename))
                        logger.info(
                            f"Saved best anti-spoof model (val_loss: {val_loss:.4f}, val_accuracy: {val_accuracy:.2f}%)")
                    except Exception as save_error:
                        logger.error(f"Error saving model: {save_error}")
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
            try:
                model_filename = "rawnet2_antispoof.pt" if is_rawnet else "anti_spoof_model.pt"
                src_path = os.path.join(self.output_path, model_filename)
                dst_path = os.path.join(self.model_path, model_filename)

                if os.path.exists(src_path):
                    shutil.copy(src_path, dst_path)
                    logger.info(f"Copied best model to {dst_path}")
                else:
                    logger.warning(f"Best model file not found: {src_path}")
            except Exception as copy_error:
                logger.error(f"Error copying model: {copy_error}")

            # Создание конфигурационного файла
            config = {
                "model_type": "RawNet2AntiSpoofing" if is_rawnet else "ImprovedAntiSpoofingNet",
                "sample_rate": self.sample_rate,
                "trained_date": datetime.now().isoformat(),
                "best_val_loss": best_val_loss,
                "best_val_accuracy": val_accuracy if 'val_accuracy' in locals() else 0.0,
                "use_pretrained_rawnet": self.use_pretrained_rawnet,
                "epochs_completed": epoch + 1,
                "early_stopped": patience_counter >= patience
            }

            try:
                with open(os.path.join(self.model_path, "anti_spoof_config.json"), "w") as f:
                    json.dump(config, f, indent=4)
                logger.info("Configuration saved successfully")
            except Exception as config_error:
                logger.error(f"Error saving configuration: {config_error}")

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


# Для обратной совместимости
RawNet2AntiSpoofing = RawNet2AntiSpoofing