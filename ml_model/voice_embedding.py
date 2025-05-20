import os
import torch
import numpy as np
import logging
import json
from pathlib import Path
import torchaudio
import torch.nn.functional as F

# Дополнительные библиотеки для улучшенной обработки аудио
import librosa
from scipy.signal import butter, filtfilt

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("voice_embedding")


class VoiceEmbeddingModel:
    """
    Улучшенный класс для извлечения голосовых эмбеддингов с улучшенной обработкой аудио
    и более надежным алгоритмом сравнения
    """

    def __init__(self, model_path=None, device=None):
        # Инициализируем устройство
        self.device = self._initialize_device(device)
        logger.info(f"Using device: {self.device}")

        # Параметры для обработки аудио
        self.sample_rate = 16000
        self.n_mels = 80
        self.emb_dim = 192
        self.target_length = 501
        self.target_feature_length = 501

        # Параметры для улучшенного сравнения
        self.default_threshold = 0.4  # Снижаем порог для лучшего обнаружения
        self.high_confidence_threshold = 0.65  # Порог высокой уверенности

        # Новые параметры для улучшенной обработки аудио
        self.bandpass_low = 200  # Нижняя граница полосы пропускания (Гц)
        self.bandpass_high = 3500  # Верхняя граница полосы пропускания (Гц)
        self.silence_threshold = 0.01  # Порог для обнаружения тишины
        self.dynamic_range_db = 30  # Динамический диапазон для нормализации

        # Создаем директорию модели, если она не существует
        if model_path:
            os.makedirs(model_path, exist_ok=True)

        # Создаем feature_extractor на этапе инициализации
        self.feature_extractor = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.sample_rate,
            n_fft=512,
            win_length=400,
            hop_length=160,
            n_mels=self.n_mels,
            f_min=20,
            f_max=7600,
            window_fn=torch.hamming_window
        ).to(self.device)

        # Инициализация модели на основе доступных библиотек
        try:
            # Пробуем использовать SpeechBrain
            logger.info("Trying to use SpeechBrain for voice embeddings")

            # Выбираем правильный путь импорта в зависимости от версии SpeechBrain
            try:
                from speechbrain.inference.speaker import EncoderClassifier
                self.encoder_class = EncoderClassifier
                logger.info("Using speechbrain.inference.speaker.EncoderClassifier")
            except ImportError:
                try:
                    from speechbrain.pretrained import EncoderClassifier
                    self.encoder_class = EncoderClassifier
                    logger.info("Using speechbrain.pretrained.EncoderClassifier")
                except ImportError:
                    raise ImportError("SpeechBrain is not properly installed")

            # Инициализация модели SpeechBrain
            self._init_speechbrain_model(model_path)
            self.using_speechbrain = True

        except Exception as e:
            # Если SpeechBrain не доступен, используем оригинальную модель
            logger.warning(f"Could not initialize SpeechBrain: {e}")
            logger.info("Falling back to original ECAPA-TDNN implementation")

            try:
                # Импортируем напрямую из SpeechBrain
                from speechbrain.lobes.models.ECAPA_TDNN import ECAPA_TDNN

                # Создание модели
                self.model = ECAPA_TDNN(
                    input_size=self.n_mels,
                    channels=[512, 512, 512, 512, 1536],
                    kernel_sizes=[5, 3, 3, 3, 1],
                    dilations=[1, 2, 3, 4, 1],
                    attention_channels=128,
                    lin_neurons=self.emb_dim
                ).to(self.device)

                # Пробуем загрузить веса, если они есть
                if model_path:
                    model_file = os.path.join(model_path, "ecapa_tdnn.pt")
                    if os.path.exists(model_file):
                        self.model.load_state_dict(torch.load(model_file, map_location=self.device))
                        logger.info(f"Successfully loaded ECAPA-TDNN model from {model_file}")

                self.model.eval()  # Переводим модель в режим оценки
                self.using_speechbrain = False

            except Exception as fallback_error:
                logger.error(f"Error initializing fallback model: {fallback_error}")
                # Создаем заглушку для метода extract_embedding
                self.using_speechbrain = False
                self.model = None  # Будем проверять это в extract_embedding

        logger.info("Voice embedding model initialized")

    def _initialize_device(self, requested_device):
        """
        Безопасно инициализирует устройство для вычислений с проверкой CUDA
        """
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

    def _init_speechbrain_model(self, model_path):
        """
        Инициализирует предобученную модель SpeechBrain ECAPA-TDNN
        """
        # Определяем пути для сохранения модели
        if model_path:
            save_dir = model_path
        else:
            save_dir = "pretrained_models/spkrec-ecapa-voxceleb"

        # Создаем директорию, если она не существует
        os.makedirs(save_dir, exist_ok=True)
        logger.info(f"SpeechBrain model will be saved in: {save_dir}")

        # Загружаем модель с huggingface или используем локальную
        try:
            self.sb_model = self.encoder_class.from_hparams(
                source="speechbrain/spkrec-ecapa-voxceleb",
                savedir=save_dir,
                run_opts={"device": self.device}
            )
            logger.info("SpeechBrain model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading SpeechBrain model: {e}")
            raise

    def _apply_bandpass_filter(self, waveform, sr):
        """
        Применяет полосовой фильтр для улучшения качества голоса
        и удаления нежелательных частот
        """
        try:
            # Создаем фильтр Баттерворта
            nyquist = 0.5 * sr
            low = self.bandpass_low / nyquist
            high = self.bandpass_high / nyquist
            b, a = butter(4, [low, high], btype='band')

            # Применяем фильтр к сигналу
            filtered_waveform = filtfilt(b, a, waveform)
            return filtered_waveform
        except Exception as e:
            logger.warning(f"Error applying bandpass filter: {e}")
            return waveform  # Возвращаем оригинальный сигнал в случае ошибки

    def _enhance_audio(self, waveform, sr):
        """
        Улучшает качество аудио для более надежного извлечения признаков
        """
        try:
            # Применяем фильтр шума с помощью спектрального вычитания
            D = librosa.stft(waveform)
            S_db = librosa.amplitude_to_db(np.abs(D))

            # Оценка уровня шума из первых кадров (обычно тишина)
            noise_frames = min(int(0.1 * S_db.shape[1]), 10)  # 10% или 10 кадров
            noise_estimate = np.mean(S_db[:, :noise_frames], axis=1, keepdims=True)

            # Применяем спектральное вычитание с сохранением динамического диапазона
            S_db_enhanced = np.maximum(S_db - noise_estimate, S_db - self.dynamic_range_db)

            # Преобразуем обратно в временную область
            S_enhanced = librosa.db_to_amplitude(S_db_enhanced)
            D_enhanced = S_enhanced * np.exp(1j * np.angle(D))
            enhanced_waveform = librosa.istft(D_enhanced, length=len(waveform))

            # Применяем дополнительно полосовой фильтр
            enhanced_waveform = self._apply_bandpass_filter(enhanced_waveform, sr)

            # Применяем компрессию динамического диапазона для улучшения SNR
            enhanced_waveform = np.sign(enhanced_waveform) * (np.abs(enhanced_waveform) ** 0.8)

            # Нормализуем амплитуду
            return enhanced_waveform / (np.max(np.abs(enhanced_waveform)) + 1e-6)
        except Exception as e:
            logger.warning(f"Error enhancing audio: {e}, returning original waveform")
            return waveform

    def _select_best_segment(self, waveform, sr):
        """
        Выбирает лучший сегмент аудио для извлечения эмбеддинга
        """
        try:
            # Применяем обнаружение речевой активности (VAD)
            # Простой VAD на основе энергии
            frame_length = int(0.025 * sr)  # 25 мс
            hop_length = int(0.010 * sr)  # 10 мс

            # Рассчитываем энергию для каждого фрейма
            energy = librosa.feature.rms(
                y=waveform,
                frame_length=frame_length,
                hop_length=hop_length
            )[0]

            # Нормализуем энергию
            energy_norm = energy / (np.max(energy) + 1e-6)

            # Определяем активные фреймы (содержащие речь)
            speech_frames = energy_norm > self.silence_threshold

            # Если нет активных фреймов, возвращаем исходный сигнал
            if not np.any(speech_frames):
                return waveform

            # Находим самый длинный непрерывный сегмент речи
            speech_runs = []
            current_run = []

            for i, is_speech in enumerate(speech_frames):
                if is_speech:
                    current_run.append(i)
                elif current_run:
                    speech_runs.append(current_run)
                    current_run = []

            if current_run:  # Добавляем последний сегмент, если он есть
                speech_runs.append(current_run)

            if not speech_runs:
                return waveform

            # Выбираем самый длинный сегмент речи
            longest_run = max(speech_runs, key=len)

            # Преобразуем индексы фреймов в индексы сэмплов
            start_sample = max(0, longest_run[0] * hop_length - frame_length)
            end_sample = min(len(waveform), (longest_run[-1] + 1) * hop_length + frame_length)

            # Проверка минимальной длины сегмента (1 секунда)
            min_segment_length = sr
            if end_sample - start_sample < min_segment_length:
                # Если сегмент слишком короткий, расширяем его
                padding = (min_segment_length - (end_sample - start_sample)) // 2
                start_sample = max(0, start_sample - padding)
                end_sample = min(len(waveform), end_sample + padding)

            # Вырезаем выбранный сегмент
            selected_segment = waveform[start_sample:end_sample]

            return selected_segment

        except Exception as e:
            logger.warning(f"Error selecting best segment: {e}, returning original waveform")
            return waveform

    def extract_embedding(self, audio_path):
        """
        Улучшенное извлечение эмбеддинга из аудиофайла - основной метод интерфейса
        с улучшенной обработкой разных форматов и повышенной надежностью
        """
        # Проверка существования файла
        if not os.path.exists(audio_path):
            logger.error(f"Audio file not found: {audio_path}")
            return None

        logger.info(f"Loading audio from: {audio_path}")

        try:
            # Используем улучшенную обработку аудио
            if self.using_speechbrain:
                return self._extract_embedding_with_enhanced_audio(audio_path)
            elif self.model is not None:
                # Иначе используем оригинальную модель, если она инициализирована
                return self._extract_embedding_original_with_enhanced_audio(audio_path)
            else:
                # Если обе модели недоступны, создаем случайный эмбеддинг для тестирования
                logger.warning("No model available, returning random embedding for testing")
                embedding = np.random.randn(192).astype(np.float32)
                # Нормализуем эмбеддинг
                embedding = embedding / np.linalg.norm(embedding)
                return embedding
        except Exception as e:
            logger.error(f"Error in extract_embedding: {e}")
            # Возвращаем случайный эмбеддинг при ошибке для тестирования
            embedding = np.random.randn(192).astype(np.float32)
            embedding = embedding / np.linalg.norm(embedding)
            return embedding

    def _extract_embedding_with_enhanced_audio(self, audio_path):
        """
        Улучшенное извлечение эмбеддинга с помощью SpeechBrain
        с расширенной обработкой аудио для повышения качества
        """
        try:
            # Сначала пробуем загрузить через librosa для поддержки большего числа форматов
            import librosa
            try:
                # Загрузка и нормализация аудио
                waveform, sr = librosa.load(audio_path, sr=self.sample_rate, mono=True)

                # Проверка на тишину или слишком короткий файл
                if len(waveform) < 0.5 * self.sample_rate or np.max(np.abs(waveform)) < 0.01:
                    logger.warning(f"Audio file {audio_path} is too short or contains silence")
                    # Генерируем синтетический сигнал для тестирования
                    waveform = np.sin(np.linspace(0, 100 * np.pi, self.sample_rate))

                # УЛУЧШЕНИЕ 1: Выбор лучшего сегмента аудио
                waveform = self._select_best_segment(waveform, sr)

                # УЛУЧШЕНИЕ 2: Улучшение качества аудио
                waveform = self._enhance_audio(waveform, sr)

                # Нормализация амплитуды
                waveform = waveform / (np.max(np.abs(waveform)) + 1e-10)

                # Преобразование в тензор PyTorch
                waveform_tensor = torch.FloatTensor(waveform).unsqueeze(0).to(self.device)

                # УЛУЧШЕНИЕ 3: Извлечение нескольких эмбеддингов из разных частей аудио и усреднение
                embeddings = []

                # 1. Используем оригинальный сегмент
                with torch.no_grad():
                    emb1 = self.sb_model.encode_batch(waveform_tensor)
                    embeddings.append(emb1.squeeze().cpu().numpy())

                # 2. Если достаточно длинное аудио, используем первую половину
                if len(waveform) > self.sample_rate * 2:
                    half_len = len(waveform) // 2
                    waveform_first_half = waveform[:half_len]
                    waveform_tensor = torch.FloatTensor(waveform_first_half).unsqueeze(0).to(self.device)

                    with torch.no_grad():
                        emb2 = self.sb_model.encode_batch(waveform_tensor)
                        embeddings.append(emb2.squeeze().cpu().numpy())

                    # 3. Используем вторую половину
                    waveform_second_half = waveform[half_len:]
                    waveform_tensor = torch.FloatTensor(waveform_second_half).unsqueeze(0).to(self.device)

                    with torch.no_grad():
                        emb3 = self.sb_model.encode_batch(waveform_tensor)
                        embeddings.append(emb3.squeeze().cpu().numpy())

                # Усредняем все полученные эмбеддинги
                if len(embeddings) > 1:
                    embedding_np = np.mean(embeddings, axis=0)
                else:
                    embedding_np = embeddings[0]

                # Проверка на NaN и Inf
                if np.isnan(embedding_np).any() or np.isinf(embedding_np).any():
                    logger.warning(f"Embedding contains NaN or Inf values in {audio_path}")
                    embedding_np = np.random.randn(self.emb_dim).astype(np.float32)
                    embedding_np = embedding_np / np.linalg.norm(embedding_np)

                # УЛУЧШЕНИЕ 4: Нормализация эмбеддинга для более стабильного сравнения
                embedding_np = embedding_np / (np.linalg.norm(embedding_np) + 1e-10)

                return embedding_np

            except Exception as librosa_error:
                logger.warning(f"Librosa loading failed: {librosa_error}, trying SpeechBrain loader")

                # Если не удалось через librosa, пробуем через SpeechBrain
                with torch.no_grad():
                    try:
                        waveform = self.sb_model.load_audio(audio_path)
                        batch = waveform.unsqueeze(0).to(self.device)
                        embedding = self.sb_model.encode_batch(batch)
                        embedding_np = embedding.squeeze().cpu().numpy()

                        # Проверка на NaN и Inf
                        if np.isnan(embedding_np).any() or np.isinf(embedding_np).any():
                            logger.warning(f"Embedding contains NaN or Inf values in {audio_path}")
                            embedding_np = np.random.randn(self.emb_dim).astype(np.float32)
                            embedding_np = embedding_np / np.linalg.norm(embedding_np)

                        return embedding_np

                    except Exception as sb_error:
                        logger.error(f"SpeechBrain audio loading failed: {sb_error}")
                        raise

        except Exception as e:
            logger.error(f"Error in SpeechBrain embedding extraction: {e}")
            # В случае ошибки генерируем случайный эмбеддинг для тестирования
            embedding = np.random.randn(192).astype(np.float32)
            embedding = embedding / np.linalg.norm(embedding)
            return embedding

    def _extract_embedding_original_with_enhanced_audio(self, audio_path):
        """
        Оригинальный метод извлечения эмбеддинга с улучшенной обработкой аудиофайлов
        """
        try:
            # Загрузка и предобработка аудио с использованием librosa
            import librosa

            # Загрузка аудио
            waveform, sr = librosa.load(audio_path, sr=self.sample_rate, mono=True)

            # Проверка на тишину или слишком короткий файл
            if len(waveform) < 0.5 * self.sample_rate or np.max(np.abs(waveform)) < 0.01:
                logger.warning(f"Audio file {audio_path} is too short or contains silence")
                # Генерируем синтетический сигнал для тестирования
                waveform = np.sin(np.linspace(0, 100 * np.pi, self.sample_rate))

            # УЛУЧШЕНИЕ 1: Выбор лучшего сегмента аудио
            waveform = self._select_best_segment(waveform, sr)

            # УЛУЧШЕНИЕ 2: Улучшение качества аудио
            waveform = self._enhance_audio(waveform, sr)

            # Нормализация
            waveform = waveform / (np.max(np.abs(waveform)) + 1e-10)

            # Обрезаем тишину, если она есть
            waveform_trimmed, _ = librosa.effects.trim(waveform, top_db=20)

            # Если после обрезки осталось достаточно аудио, используем обрезанное
            if len(waveform_trimmed) >= 0.5 * self.sample_rate:
                waveform = waveform_trimmed

            # Стандартизация длины для обработки
            if len(waveform) > self.sample_rate * 10:  # Если длиннее 10 секунд
                # Берем 10 секунд из середины для более стабильного результата
                center = len(waveform) // 2
                start = center - (self.sample_rate * 5)
                end = center + (self.sample_rate * 5)
                waveform = waveform[max(0, start):min(len(waveform), end)]
            elif len(waveform) < self.sample_rate:  # Если короче 1 секунды
                # Повторяем аудио до достижения 1 секунды
                repeats = int(np.ceil(self.sample_rate / len(waveform)))
                waveform = np.tile(waveform, repeats)[:self.sample_rate]

            # УЛУЧШЕНИЕ 3: Извлечение нескольких эмбеддингов из разных частей аудио и усреднение
            embeddings = []

            # Подготовка основного сегмента
            waveform_tensor = torch.FloatTensor(waveform).unsqueeze(0).to(self.device)
            mel_spectrogram = self.feature_extractor(waveform_tensor)
            log_mel = torch.log(mel_spectrogram + 1e-9)
            mean, std = -4.5, 2.0  # Стандартные значения для нормализации
            normalized_features = (log_mel - mean) / std

            # Подготовка правильных размерностей
            if normalized_features.shape[2] != self.target_feature_length:
                normalized_features = F.interpolate(
                    normalized_features,
                    size=self.target_feature_length,
                    mode='linear',
                    align_corners=False
                )

            # Основной эмбеддинг
            with torch.no_grad():
                if normalized_features.shape[1] == self.n_mels:
                    features = normalized_features.transpose(1, 2)  # [B, F, T] -> [B, T, F]
                else:
                    features = normalized_features

                # Вызов модели
                embedding = self.model(features)
                # Нормализация
                embedding = F.normalize(embedding, p=2, dim=1)
                embeddings.append(embedding.squeeze().cpu().numpy())

            # Если аудио достаточно длинное, извлечем дополнительные эмбеддинги
            if len(waveform) > self.sample_rate * 2:
                # Делим на сегменты
                segments = []
                segment_length = self.sample_rate
                for i in range(0, len(waveform) - segment_length, segment_length):
                    segments.append(waveform[i:i + segment_length])

                # Используем до 3 дополнительных сегментов
                for i, segment in enumerate(segments[:3]):
                    waveform_tensor = torch.FloatTensor(segment).unsqueeze(0).to(self.device)
                    mel_spectrogram = self.feature_extractor(waveform_tensor)
                    log_mel = torch.log(mel_spectrogram + 1e-9)
                    normalized_features = (log_mel - mean) / std

                    if normalized_features.shape[2] != self.target_feature_length:
                        normalized_features = F.interpolate(
                            normalized_features,
                            size=self.target_feature_length,
                            mode='linear',
                            align_corners=False
                        )

                    with torch.no_grad():
                        if normalized_features.shape[1] == self.n_mels:
                            features = normalized_features.transpose(1, 2)
                        else:
                            features = normalized_features

                        segment_embedding = self.model(features)
                        segment_embedding = F.normalize(segment_embedding, p=2, dim=1)
                        embeddings.append(segment_embedding.squeeze().cpu().numpy())

            # Усредняем все эмбеддинги
            if len(embeddings) > 1:
                embedding_np = np.mean(embeddings, axis=0)
            else:
                embedding_np = embeddings[0]

            # Проверка на NaN и Inf
            if np.isnan(embedding_np).any() or np.isinf(embedding_np).any():
                logger.warning(f"Embedding contains NaN or Inf values in {audio_path}")
                embedding_np = np.random.randn(self.emb_dim).astype(np.float32)
                embedding_np = embedding_np / np.linalg.norm(embedding_np)

            # УЛУЧШЕНИЕ 4: Нормализация эмбеддинга для более стабильного сравнения
            embedding_np = embedding_np / (np.linalg.norm(embedding_np) + 1e-10)

            return embedding_np

        except Exception as e:
            logger.error(f"Error in original embedding extraction: {e}")
            # В случае ошибки генерируем случайный эмбеддинг для тестирования
            embedding = np.random.randn(self.emb_dim).astype(np.float32)
            embedding = embedding / np.linalg.norm(embedding)
            return embedding

    def improved_compare_embeddings(self, embedding1, embedding2, threshold=None):
        """
        Улучшенное сравнение двух эмбеддингов для определения схожести голосов.
        Использует комбинацию нескольких метрик для более надежного сравнения.

        Args:
            embedding1: Первый эмбеддинг (numpy массив)
            embedding2: Второй эмбеддинг (numpy массив)
            threshold: Пороговое значение (если None, используется self.default_threshold)

        Returns:
            tuple: (similarity, is_match) - Значение схожести [0-1] и булево (совпадение найдено)
        """
        try:
            # Используем заданный порог или значение по умолчанию
            if threshold is None:
                threshold = self.default_threshold

            # Проверка на None или невалидные значения
            if embedding1 is None or embedding2 is None:
                return 0.0, False

            if np.isnan(embedding1).any() or np.isnan(embedding2).any():
                return 0.0, False

            # Нормализация векторов для косинусного сходства
            embedding1_norm = embedding1 / (np.linalg.norm(embedding1) + 1e-10)
            embedding2_norm = embedding2 / (np.linalg.norm(embedding2) + 1e-10)

            # 1. Косинусное сходство - базовая метрика
            cosine_similarity = np.dot(embedding1_norm, embedding2_norm)

            # 2. Евклидово расстояние (нормализованное)
            euclidean_distance = np.linalg.norm(embedding1_norm - embedding2_norm)
            euclidean_similarity = 1.0 / (1.0 + euclidean_distance)  # Преобразуем в меру сходства

            # 3. Манхэттенское расстояние (нормализованное)
            manhattan_distance = np.sum(np.abs(embedding1_norm - embedding2_norm))
            manhattan_similarity = 1.0 / (1.0 + manhattan_distance)  # Преобразуем в меру сходства

            # Объединение метрик с весами
            # Косинусное сходство имеет наибольший вес, так как более стабильно
            weighted_similarity = (
                    0.7 * cosine_similarity +
                    0.2 * euclidean_similarity +
                    0.1 * manhattan_similarity
            )

            # Преобразование из диапазона [-1, 1] в [0, 1]
            normalized_similarity = (weighted_similarity + 1) / 2

            # Применяем сигмоидную функцию для усиления контраста между
            # совпадающими и несовпадающими эмбеддингами
            def sigmoid_scale(x, center=0.7, sharpness=8.0):
                return 1.0 / (1.0 + np.exp(-sharpness * (x - center)))

            # Применяем сигмоидное масштабирование только если базовое сходство выше 0.3
            # чтобы избежать ложных срабатываний при случайных совпадениях
            final_similarity = normalized_similarity
            if normalized_similarity > 0.3:
                final_similarity = sigmoid_scale(normalized_similarity)

            # Определяем, является ли это совпадением
            is_match = final_similarity >= threshold

            # Добавляем бонус для очень близких совпадений
            if final_similarity > 0.9:
                final_similarity = min(1.0, final_similarity + 0.05)

            return final_similarity, is_match

        except Exception as e:
            logger.error(f"Error comparing embeddings: {e}")
            return 0.0, False

    def compare_embeddings_batch(self, reference_embeddings, query_embedding, threshold=None):
        """
        Сравнивает один запрос со всеми эталонными эмбеддингами.
        Использует более надежную стратегию голосования для принятия решения.

        Args:
            reference_embeddings: Список эталонных эмбеддингов
            query_embedding: Эмбеддинг запроса
            threshold: Пороговое значение (если None, используется self.default_threshold)

        Returns:
            dict: Результаты сравнения с метриками
        """
        try:
            if threshold is None:
                threshold = self.default_threshold

            if not reference_embeddings or query_embedding is None:
                return {
                    "match_found": False,
                    "similarity": 0.0,
                    "confidence": 0.0,
                    "voting_score": 0.0,
                    "matching_embeddings": 0
                }

            # Сравниваем со всеми эталонными эмбеддингами
            similarities = []
            matches = []

            for ref_emb in reference_embeddings:
                sim, is_match = self.improved_compare_embeddings(
                    query_embedding, ref_emb, threshold
                )
                similarities.append(sim)
                matches.append(is_match)

            # Находим максимальное сходство
            max_similarity = max(similarities) if similarities else 0.0

            # Определяем количество совпадений
            match_count = sum(matches)

            # Вычисляем показатель голосования
            total_embeddings = len(reference_embeddings)
            voting_score = match_count / total_embeddings if total_embeddings > 0 else 0.0

            # Определяем уровень уверенности на основе
            # 1. максимального сходства
            # 2. количества совпадений
            # 3. консистентности сходства (низкая дисперсия = высокая уверенность)
            similarity_variance = np.var(similarities) if len(similarities) > 1 else 0.0
            variance_penalty = min(0.3, similarity_variance * 3.0)  # Штраф за высокую дисперсию

            # Бонус за количество совпадений
            match_ratio = match_count / total_embeddings if total_embeddings > 0 else 0.0
            match_bonus = match_ratio * 0.3  # До 30% бонуса за полное совпадение

            # Итоговая уверенность
            confidence = max(0.0, min(1.0, max_similarity - variance_penalty + match_bonus))

            # Принятие решения о совпадении
            # Используем комбинацию голосования и максимального сходства
            match_found = False

            # Стратегия 1: По абсолютному количеству совпадений (если достаточно образцов)
            if total_embeddings >= 3 and match_count >= max(2, total_embeddings // 2):
                match_found = True
            # Стратегия 2: По максимальному сходству с высокой уверенностью
            elif max_similarity >= self.high_confidence_threshold:
                match_found = True
            # Стратегия 3: Комбинированная для небольшого количества образцов
            elif total_embeddings < 3 and max_similarity >= threshold and confidence > 0.6:
                match_found = True

            return {
                "match_found": match_found,
                "similarity": float(max_similarity),
                "confidence": float(confidence),
                "voting_score": float(voting_score),
                "matching_embeddings": match_count
            }

        except Exception as e:
            logger.error(f"Error in batch comparison: {e}")
            return {
                "match_found": False,
                "similarity": 0.0,
                "confidence": 0.0,
                "voting_score": 0.0,
                "matching_embeddings": 0,
                "error": str(e)
            }

    def save_embeddings(self, embeddings_dict, output_file):
        """
        Сохранение эмбеддингов в файл

        Args:
            embeddings_dict: Словарь {user_id: [embedding1, embedding2, ...]}
            output_file: Путь к файлу для сохранения
        """
        try:
            # Подготовка директории
            output_dir = os.path.dirname(output_file)
            os.makedirs(output_dir, exist_ok=True)

            # Преобразование numpy массивов в списки для JSON
            json_compatible = {}
            for user_id, embeddings in embeddings_dict.items():
                json_compatible[user_id] = [emb.tolist() for emb in embeddings]

            # Сохранение в JSON
            with open(output_file, 'w') as f:
                json.dump(json_compatible, f)

            logger.info(f"Embeddings saved to {output_file}")
            return True

        except Exception as e:
            logger.error(f"Error saving embeddings: {e}")
            return False

    def load_embeddings(self, input_file):
        """
        Загрузка эмбеддингов из файла

        Args:
            input_file: Путь к файлу с эмбеддингами

        Returns:
            embeddings_dict: Словарь {user_id: [embedding1, embedding2, ...]}
        """
        try:
            # Проверка существования файла
            if not os.path.exists(input_file):
                logger.error(f"Embeddings file not found: {input_file}")
                return {}

            # Загрузка из JSON
            with open(input_file, 'r') as f:
                data = json.load(f)

            # Преобразование списков обратно в numpy массивы
            embeddings_dict = {}
            for user_id, embeddings in data.items():
                embeddings_dict[user_id] = [np.array(emb) for emb in embeddings]

            logger.info(f"Loaded embeddings for {len(embeddings_dict)} users")
            return embeddings_dict

        except Exception as e:
            logger.error(f"Error loading embeddings: {e}")
            return {}
        try:
            # Проверка существования файла
            if not os.path.exists(input_file):
                logger.error(f"Embeddings file not found: {input_file}")
                return {}

            # Загрузка из JSON
            with open(input_file, 'r') as f:
                data = json.load(f)

            # Преобразование списков обратно в numpy массивы
            embeddings_dict = {}
            for user_id, embeddings in data.items():
                embeddings_dict[user_id] = [np.array(emb) for emb in embeddings]

            logger.info(f"Loaded embeddings for {len(embeddings_dict)} users")
            return embeddings_dict

        except Exception as e:
            logger.error(f"Error load embeddings: {e}")