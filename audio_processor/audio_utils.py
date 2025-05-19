# audio_processor/audio_utils.py
import os
import numpy as np
import torch
import torchaudio
import torchaudio.functional as F
import torchaudio.transforms as T
from speechbrain.pretrained import EncoderClassifier
import logging
from pathlib import Path
import soundfile as sf

logger = logging.getLogger(__name__)


class AudioProcessor:
    def __init__(self, model_path=None, device=None, sample_rate=16000, target_duration=3.0):
        """
        Инициализация аудио-процессора с моделью голосовой биометрии

        Args:
            model_path: Путь к локальной модели (опционально)
            device: Устройство для вычислений ('cuda', 'cpu')
            sample_rate: Целевая частота дискретизации
            target_duration: Целевая длительность аудио в секундах
        """
        # Настройка устройства вычислений с обработкой ошибок
        if device is None:
            # Попытка использовать CUDA если доступна
            if torch.cuda.is_available():
                try:
                    # Проверка работоспособности CUDA
                    test_tensor = torch.tensor([1.0], device='cuda')
                    _ = test_tensor * 2  # Тестовая операция
                    self.device = 'cuda'
                except Exception as e:
                    logger.warning(f"CUDA available but failed: {e}. Falling back to CPU.")
                    self.device = 'cpu'
            else:
                self.device = 'cpu'
        else:
            self.device = device

        logger.info(f"Using device: {self.device}")

        # Параметры предобработки аудио
        self.sample_rate = sample_rate
        self.target_length = int(target_duration * self.sample_rate)  # Целевая длина в сэмплах

        # Загрузить предобученную модель ECAPA-TDNN из SpeechBrain
        try:
            if model_path and os.path.exists(model_path):
                logger.info(f"Loading model from local path: {model_path}")
                self.model = EncoderClassifier.from_hparams(
                    source=model_path,
                    run_opts={"device": self.device}
                )
            else:
                logger.info("Loading model from HuggingFace")
                self.model = EncoderClassifier.from_hparams(
                    source="speechbrain/spkrec-ecapa-voxceleb",
                    savedir="pretrained_models/spkrec-ecapa-voxceleb",
                    run_opts={"device": self.device}
                )
            logger.info("Model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise RuntimeError(f"Failed to load voice embedding model: {e}")

    def preprocess_audio(self, audio_path):
        """
        Предобработка аудиофайла: ресемплирование, нормализация, нарезка

        Args:
            audio_path: Путь к аудиофайлу

        Returns:
            torch.Tensor: Предобработанный аудио-сигнал
        """
        logger.debug(f"Preprocessing audio: {audio_path}")

        # Проверка существования файла
        if not os.path.exists(audio_path):
            error_msg = f"Audio file not found: {audio_path}"
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)

        try:
            # Загрузка аудио через torchaudio
            waveform, sample_rate = torchaudio.load(audio_path)
        except Exception as e:
            logger.warning(f"Error loading audio with torchaudio: {e}, trying alternative methods")
            try:
                # Пробуем альтернативную загрузку через soundfile
                audio_data, sample_rate = sf.read(audio_path)
                # Конвертация в формат torchaudio
                if audio_data.ndim == 1:  # Моно
                    audio_data = audio_data.reshape(1, -1)
                else:  # Стерео или многоканальное
                    audio_data = audio_data.T
                waveform = torch.tensor(audio_data, dtype=torch.float32)
            except Exception as alt_e:
                logger.error(f"All loading methods failed: {alt_e}")
                raise RuntimeError(f"Failed to load audio file {audio_path}: {e}, {alt_e}")

        # Проверка на наличие данных
        if waveform.numel() == 0:
            error_msg = f"Audio file {audio_path} contains no data"
            logger.error(error_msg)
            raise ValueError(error_msg)

        # Преобразование в моно если стерео
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        # Ресемплирование если необходимо
        if sample_rate != self.sample_rate:
            logger.debug(f"Resampling from {sample_rate} to {self.sample_rate}")
            try:
                waveform = F.resample(waveform, sample_rate, self.sample_rate)
            except Exception as e:
                logger.error(f"Resampling failed: {e}")
                raise RuntimeError(f"Failed to resample audio: {e}")

        # Удаление тишины
        with torch.no_grad():
            # Вычисление энергии сигнала
            energy = torch.sqrt(torch.mean(waveform ** 2, dim=1))
            if energy.max() < 0.001:
                logger.warning(f"Audio file {audio_path} seems to be silent")

        # Нормализация громкости
        waveform = F.gain(waveform, 3.0)

        # Обрезка или дополнение до целевой длины
        if waveform.shape[1] > self.target_length:
            # Берем средний фрагмент для получения наиболее информативной части
            start = (waveform.shape[1] - self.target_length) // 2
            waveform = waveform[:, start:start + self.target_length]
            logger.debug(f"Audio trimmed to target length: {self.target_length} samples")
        else:
            # Дополняем нулями если аудио слишком короткое
            padding = self.target_length - waveform.shape[1]
            waveform = torch.nn.functional.pad(waveform, (0, padding))
            logger.debug(f"Audio padded to target length: {self.target_length} samples")

        return waveform

    def extract_embedding(self, audio_path):
        """
        Извлечение эмбеддинга голоса из аудиофайла

        Args:
            audio_path: Путь к аудиофайлу

        Returns:
            numpy.ndarray: Векторное представление голоса
        """
        try:
            # Предобработка аудио
            waveform = self.preprocess_audio(audio_path)

            # Перенос данных на нужное устройство
            try:
                waveform = waveform.to(self.device)
            except Exception as e:
                logger.error(f"Error moving data to device {self.device}: {e}")
                logger.info("Falling back to CPU")
                self.device = 'cpu'
                waveform = waveform.to(self.device)

            # Извлечение эмбеддинга
            with torch.no_grad():
                try:
                    embeddings = self.model.encode_batch(waveform)
                    embedding = embeddings.squeeze().cpu().numpy()
                    logger.debug(f"Embedding extracted, shape: {embedding.shape}")
                    return embedding
                except Exception as e:
                    logger.error(f"Error extracting embedding: {e}")
                    # Попробуем обойти проблемы с совместимостью формата данных
                    if "expected 2D tensor" in str(e) or "expected 3D tensor" in str(e):
                        # Изменяем размерность если необходимо
                        if waveform.dim() == 2:
                            waveform = waveform.unsqueeze(0)  # Добавляем размерность батча
                        embeddings = self.model.encode_batch(waveform)
                        embedding = embeddings.squeeze().cpu().numpy()
                        logger.debug(f"Embedding extracted with shape correction, shape: {embedding.shape}")
                        return embedding
                    else:
                        raise RuntimeError(f"Failed to extract embedding: {e}")
        except Exception as e:
            logger.error(f"Failed to process audio {audio_path}: {e}")
            raise

    def compare_embeddings(self, embedding1, embedding2, threshold=0.75):
        """
        Сравнение двух эмбеддингов с использованием косинусного сходства

        Args:
            embedding1: Первый эмбеддинг
            embedding2: Второй эмбеддинг
            threshold: Порог сходства для определения совпадения (0.0-1.0)

        Returns:
            tuple: (сходство (0.0-1.0), результат сравнения (True/False))
        """
        try:
            # Проверка размерностей
            if embedding1.shape != embedding2.shape:
                raise ValueError(f"Embedding shapes do not match: {embedding1.shape} vs {embedding2.shape}")

            # Проверка на нулевые векторы
            if np.all(embedding1 == 0) or np.all(embedding2 == 0):
                logger.warning("One of the embeddings is a zero vector")
                return 0.0, False

            # Косинусное сходство
            similarity = np.dot(embedding1, embedding2) / (np.linalg.norm(embedding1) * np.linalg.norm(embedding2))

            # Проверка на NaN (может возникнуть при делении на очень маленькие числа)
            if np.isnan(similarity):
                logger.warning("Similarity calculation resulted in NaN")
                return 0.0, False

            # Нормализация в диапазон [0, 1]
            similarity = (similarity + 1) / 2

            logger.debug(f"Similarity: {similarity:.4f}, threshold: {threshold}")
            return similarity, similarity >= threshold
        except Exception as e:
            logger.error(f"Error comparing embeddings: {e}")
            raise RuntimeError(f"Failed to compare embeddings: {e}")

    def detect_spoofing(self, audio_path, threshold=0.6):
        """
        Обнаружение спуфинг-атак в аудиофайле

        Args:
            audio_path: Путь к аудиофайлу
            threshold: Порог для определения подделки (0.0-1.0)

        Returns:
            dict: Результат анализа на спуфинг
        """
        try:
            # Предобработка аудио
            waveform = self.preprocess_audio(audio_path)

            # 1. Анализ спектрограммы
            spectrogram = T.Spectrogram()(waveform)

            # 2. Анализ мел-спектрограммы для выявления артефактов синтеза
            melspec = T.MelSpectrogram(
                sample_rate=self.sample_rate,
                n_fft=1024,
                hop_length=256,
                n_mels=80
            )(waveform)

            # 3. Анализ временного ряда для обнаружения повторяющихся паттернов (признак записи)
            # Простое вычисление автокорреляции
            signal = waveform.squeeze().cpu().numpy()
            autocorr = np.correlate(signal, signal, mode='full')
            autocorr = autocorr[len(signal) - 1:]
            normalized_autocorr = autocorr / autocorr[0]

            # Вычисление метрик для обнаружения спуфинга

            # Метрика 1: Вариативность спектра (синтезированная речь обычно менее вариативна)
            spectral_variability = torch.std(spectrogram).item()
            norm_spec_var = min(1.0, spectral_variability / 0.15)  # Нормализация

            # Метрика 2: Энергия высоких частот (часто отсутствует в синтезированной речи)
            high_freq_energy = torch.mean(spectrogram[-20:, :]).item()  # Верхние 20 бинов
            norm_high_freq = min(1.0, high_freq_energy / 0.05)  # Нормализация

            # Метрика 3: Периодичность (может указывать на повторы в записанной речи)
            # Ищем второй пик автокорреляции
            ac_peaks = []
            for i in range(1, len(normalized_autocorr) - 1):
                if (normalized_autocorr[i] > normalized_autocorr[i - 1] and
                        normalized_autocorr[i] > normalized_autocorr[i + 1] and
                        normalized_autocorr[i] > 0.2):
                    ac_peaks.append((i, normalized_autocorr[i]))

            periodicity_score = 0.0
            if len(ac_peaks) > 1:
                # Высокая периодичность может быть признаком синтеза или воспроизведения
                periodicity_score = min(1.0, ac_peaks[0][1] / 0.7)

            # Интегральный показатель подлинности
            authenticity_score = 0.5 * norm_spec_var + 0.3 * norm_high_freq + 0.2 * (1 - periodicity_score)
            authenticity_score = min(1.0, max(0.0, authenticity_score))

            return {
                "is_spoofing_detected": authenticity_score < threshold,
                "authenticity_score": authenticity_score,
                "confidence": min(abs(authenticity_score - 0.5) * 2, 0.99),
                "metrics": {
                    "spectral_variability": norm_spec_var,
                    "high_freq_energy": norm_high_freq,
                    "periodicity": periodicity_score
                }
            }
        except Exception as e:
            logger.error(f"Error in spoofing detection: {e}")
            # При ошибке возвращаем неопределенный результат с низкой уверенностью
            return {
                "is_spoofing_detected": None,
                "authenticity_score": 0.5,
                "confidence": 0.0,
                "error": str(e)
            }

    def save_embedding(self, embedding, output_path):
        """
        Сохранение эмбеддинга в файл

        Args:
            embedding: Эмбеддинг для сохранения
            output_path: Путь для сохранения
        """
        try:
            # Создаем директорию если не существует
            os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

            # Сохраняем как numpy-массив
            np.save(output_path, embedding)
            logger.info(f"Embedding saved to {output_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to save embedding: {e}")
            return False

    def load_embedding(self, input_path):
        """
        Загрузка эмбеддинга из файла

        Args:
            input_path: Путь к файлу эмбеддинга

        Returns:
            numpy.ndarray: Загруженный эмбеддинг
        """
        try:
            if not os.path.exists(input_path):
                raise FileNotFoundError(f"Embedding file not found: {input_path}")

            embedding = np.load(input_path)
            logger.info(f"Embedding loaded from {input_path}, shape: {embedding.shape}")
            return embedding
        except Exception as e:
            logger.error(f"Failed to load embedding: {e}")
            raise


# Пример использования
if __name__ == "__main__":
    # Настройка логирования
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    try:
        processor = AudioProcessor()

        # Пример обработки файла
        audio_path = "test_audio.wav"

        if not os.path.exists(audio_path):
            print(f"Warning: Test file {audio_path} not found. Creating a dummy audio file.")
            # Создаем тестовый аудиофайл если он не существует
            sample_rate = 16000
            dummy_audio = np.sin(2 * np.pi * 440 * np.arange(sample_rate * 3) / sample_rate).astype(np.float32)
            sf.write(audio_path, dummy_audio, sample_rate)

        print(f"Extracting embedding from {audio_path}")
        embedding = processor.extract_embedding(audio_path)
        print(f"Embedding shape: {embedding.shape}")

        # Сохранение эмбеддинга
        embedding_path = "test_embedding.npy"
        processor.save_embedding(embedding, embedding_path)

        # Сравнение с ранее сохраненным эмбеддингом
        stored_embedding = np.random.rand(*embedding.shape)  # В реальности это будет загруженный эмбеддинг
        similarity, is_match = processor.compare_embeddings(embedding, stored_embedding)
        print(f"Similarity: {similarity:.4f}, Is match: {is_match}")

        # Проверка на спуфинг
        spoofing_result = processor.detect_spoofing(audio_path)
        print(f"Spoofing detection result: {spoofing_result}")

    except Exception as e:
        print(f"Error in example: {e}")