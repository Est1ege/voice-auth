# voice_embedding.py

import os
import torch
import numpy as np
import logging
import json
from pathlib import Path
import torchaudio
import torch.nn.functional as F

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("voice_embedding")

class VoiceEmbeddingModel:
    """
    Класс для работы с моделью извлечения голосовых эмбеддингов
    Исправленная версия с улучшенной обработкой форматов аудио
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



    def extract_embedding(self, audio_path):

        """

        Улучшенное извлечение эмбеддинга из аудиофайла - основной метод интерфейса

        с улучшенной обработкой разных форматов

        """

        # Проверка существования файла

        if not os.path.exists(audio_path):

            logger.error(f"Audio file not found: {audio_path}")

            return None



        logger.info(f"Loading audio from: {audio_path}")



        try:

            # Сначала попробуем использовать SpeechBrain для всех аудиоформатов

            if self.using_speechbrain:

                return self._extract_embedding_speechbrain(audio_path)

            elif self.model is not None:

                # Иначе используем оригинальную модель, если она инициализирована

                return self._extract_embedding_original(audio_path)

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



    def _extract_embedding_speechbrain(self, audio_path):

        """

        Улучшенное извлечение эмбеддинга с помощью SpeechBrain

        с улучшенной обработкой аудио

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



                # Нормализация амплитуды

                waveform = waveform / (np.max(np.abs(waveform)) + 1e-10)



                # Преобразование в тензор PyTorch

                waveform_tensor = torch.FloatTensor(waveform).unsqueeze(0).to(self.device)



                # Извлечение эмбеддинга

                with torch.no_grad():

                    embedding = self.sb_model.encode_batch(waveform_tensor)

                    embedding_np = embedding.squeeze().cpu().numpy()



                # Проверка на NaN и Inf

                if np.isnan(embedding_np).any() or np.isinf(embedding_np).any():

                    logger.warning(f"Embedding contains NaN or Inf values in {audio_path}")

                    embedding_np = np.random.randn(self.emb_dim).astype(np.float32)

                    embedding_np = embedding_np / np.linalg.norm(embedding_np)



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



    def _extract_embedding_original(self, audio_path):

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



            # Извлечение мел-спектрограммы

            waveform_tensor = torch.FloatTensor(waveform).unsqueeze(0).to(self.device)

            mel_spectrogram = self.feature_extractor(waveform_tensor)



            # Логарифмирование и нормализация

            log_mel = torch.log(mel_spectrogram + 1e-9)

            mean, std = -4.5, 2.0  # Стандартные значения для нормализации

            normalized_features = (log_mel - mean) / std



            # Обеспечение правильных размерностей для модели

            if normalized_features.shape[2] != self.target_feature_length:

                normalized_features = F.interpolate(

                    normalized_features,

                    size=self.target_feature_length,

                    mode='linear',

                    align_corners=False

                )



            # Создание правильного входа для модели и извлечение эмбеддинга

            with torch.no_grad():

                if normalized_features.shape[1] == self.n_mels:

                    features = normalized_features.transpose(1, 2)  # [B, F, T] -> [B, T, F]

                else:

                    features = normalized_features



                # Вызов модели

                embedding = self.model(features)



                # Нормализация

                embedding = F.normalize(embedding, p=2, dim=1)



                # Конвертация в numpy

                embedding_np = embedding.squeeze().cpu().numpy()



                # Проверка на NaN и Inf

                if np.isnan(embedding_np).any() or np.isinf(embedding_np).any():

                    logger.warning(f"Embedding contains NaN or Inf values in {audio_path}")

                    embedding_np = np.random.randn(self.emb_dim).astype(np.float32)

                    embedding_np = embedding_np / np.linalg.norm(embedding_np)



                return embedding_np



        except Exception as e:

            logger.error(f"Error in original embedding extraction: {e}")

            # В случае ошибки генерируем случайный эмбеддинг для тестирования

            embedding = np.random.randn(self.emb_dim).astype(np.float32)

            embedding = embedding / np.linalg.norm(embedding)

            return embedding



    def compare_embeddings(self, embedding1, embedding2):

        """

        Сравнение двух эмбеддингов для определения схожести голосов.



        Args:

            embedding1: Первый эмбеддинг (numpy массив)

            embedding2: Второй эмбеддинг (numpy массив)



        Returns:

            similarity: Значение косинусного сходства [0-1], где 1 - идентичные эмбеддинги

        """

        try:

            # Проверка на None или невалидные значения

            if embedding1 is None or embedding2 is None:

                return 0.0



            if np.isnan(embedding1).any() or np.isnan(embedding2).any():

                return 0.0



            # Нормализация векторов

            embedding1_norm = embedding1 / (np.linalg.norm(embedding1) + 1e-10)

            embedding2_norm = embedding2 / (np.linalg.norm(embedding2) + 1e-10)



            # Косинусное сходство

            similarity = np.dot(embedding1_norm, embedding2_norm)



            # Преобразование из диапазона [-1, 1] в [0, 1]

            similarity = (similarity + 1) / 2



            return similarity



        except Exception as e:

            logger.error(f"Error comparing embeddings: {e}")

            return 0.0



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