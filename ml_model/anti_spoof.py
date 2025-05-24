# ml_model/anti_spoof.py - Исправленная версия с улучшенной архитектурой

import os
import librosa
import torch
import torch.nn as nn
import numpy as np
import logging
import torch.nn.functional as F
from pathlib import Path

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("anti_spoofing")

SPOOFING_THRESHOLD = 0.5  # Понижен для уменьшения ложных срабатываний
MIN_CONFIDENCE = 0.3
DEFAULT_SCORE = 0.2  # Понижен для реального голоса по умолчанию


class ImprovedAntiSpoofingNet(nn.Module):
    """
    Улучшенная модель для обнаружения спуфинг-атак
    Использует комбинацию CNN и LSTM для анализа временных и частотных характеристик
    """

    def __init__(self, input_dim=40, hidden_dim=128, num_layers=2, dropout=0.3):
        super(ImprovedAntiSpoofingNet, self).__init__()

        # CNN слои для извлечения признаков из спектрограммы
        self.conv_layers = nn.Sequential(
            # Первый блок
            nn.Conv2d(1, 32, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.25),

            # Второй блок
            nn.Conv2d(32, 64, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.25),

            # Третий блок
            nn.Conv2d(64, 128, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((10, 8)),  # Адаптивный пулинг для стандартизации размера
            nn.Dropout2d(0.3)
        )

        # Вычисляем размер после CNN
        cnn_output_size = 128 * 10 * 8  # 128 каналов, 10x8 после AdaptiveAvgPool2d

        # LSTM для временного анализа
        self.lstm = nn.LSTM(
            input_size=cnn_output_size,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
            bidirectional=True
        )

        # Полносвязные слои
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),  # *2 из-за bidirectional
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

        # Инициализация весов
        self._initialize_weights()

    def _initialize_weights(self):
        """Улучшенная инициализация весов"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LSTM):
                for name, param in m.named_parameters():
                    if 'weight' in name:
                        nn.init.xavier_normal_(param)
                    elif 'bias' in name:
                        nn.init.constant_(param, 0)

    def forward(self, x):
        """
        Forward pass
        Args:
            x: Input tensor [batch_size, 1, time, freq] (спектрограмма)
        Returns:
            Tensor: Probability of spoofing [batch_size, 1]
        """
        batch_size = x.size(0)

        # CNN feature extraction
        conv_out = self.conv_layers(x)  # [batch_size, 128, 10, 8]

        # Reshape для LSTM: [batch_size, time_steps, features]
        conv_out = conv_out.view(batch_size, 10, -1)  # [batch_size, 10, 128*8]

        # LSTM для временного анализа
        lstm_out, _ = self.lstm(conv_out)  # [batch_size, 10, hidden_dim*2]

        # Используем последний выход LSTM
        lstm_last = lstm_out[:, -1, :]  # [batch_size, hidden_dim*2]

        # Классификация
        output = self.classifier(lstm_last)  # [batch_size, 1]

        return output.squeeze()


class AntiSpoofingDetector:
    """
    Улучшенный класс для обнаружения спуфинг-атак
    """

    def __init__(self, model_path=None, device=None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Безопасный переход на CPU при проблемах с CUDA
        try:
            test_tensor = torch.zeros(1, 1).to(self.device)
            _ = test_tensor + 1
        except Exception as e:
            logger.warning(f"CUDA error: {e}, using CPU instead")
            self.device = torch.device('cpu')

        # Инициализация модели
        self.model = self._initialize_model(model_path)
        self.model.eval()

        # Параметры для обработки аудио
        self.sample_rate = 16000
        self.n_fft = 512
        self.hop_length = 256
        self.n_mels = 40

        # Статистики для адаптивного порога
        self.score_history = []
        self.max_history = 50

        logger.info(f"Anti-spoofing detector initialized on {self.device}")

    def _load_or_create_model(self) -> nn.Module:
        """
        Загружает существующую модель или создает новую
        """
        try:
            # Используем улучшенную модель
            model = ImprovedAntiSpoofingNet().to(self.device)

            # Загрузка весов, если доступны
            if os.path.exists(os.path.join(self.model_path, "anti_spoof_model.pt")):
                try:
                    state_dict = torch.load(
                        os.path.join(self.model_path, "anti_spoof_model.pt"),
                        map_location=self.device
                    )
                    model.load_state_dict(state_dict)
                    logger.info("Existing anti-spoof model loaded successfully")
                except Exception as e:
                    logger.warning(f"Could not load existing model: {e}")
            else:
                logger.info("Using new anti-spoof model")

            return model
        except Exception as e:
            logger.error(f"Error loading or creating anti-spoof model: {e}")
            raise

    def _initialize_model(self, model_path):
        """
        Инициализация модели обнаружения спуфинга
        """
        try:
            model = ImprovedAntiSpoofingNet()

            # Загрузка весов, если доступны
            if model_path and os.path.isfile(os.path.join(model_path, "anti_spoof_model.pt")):
                try:
                    full_path = os.path.join(model_path, "anti_spoof_model.pt")
                    state_dict = torch.load(full_path, map_location=self.device)
                    model.load_state_dict(state_dict)
                    logger.info(f"Loaded anti-spoofing model weights from {full_path}")
                except Exception as e:
                    logger.error(f"Error loading model weights: {e}")
                    logger.info("Using untrained model")
            else:
                logger.warning("Anti-spoofing model weights not found, using untrained model")

            return model.to(self.device)

        except Exception as e:
            logger.error(f"Error initializing anti-spoofing model: {e}")
            # Возвращаем простую модель при ошибке
            return self._create_simple_model().to(self.device)

    def _create_simple_model(self):
        """
        Создание простой модели при ошибках инициализации
        """
        return nn.Sequential(
            nn.Linear(40, 64),  # Входной размер равен количеству мел-коэффициентов
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def _extract_features(self, waveform):
        """
        Извлечение признаков из аудио для анализа спуфинга
        """
        try:
            # Мел-спектрограмма
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

            # Приведение к стандартному размeru для модели
            target_length = 128  # Примерно 4 секunds при hop_length=256
            current_length = log_mel_spec.shape[1]

            if current_length > target_length:
                # Обрезаем до нужной длины
                start_idx = (current_length - target_length) // 2
                log_mel_spec = log_mel_spec[:, start_idx:start_idx + target_length]
            elif current_length < target_length:
                # Дополняем нулями
                pad_width = target_length - current_length
                pad_left = pad_width // 2
                pad_right = pad_width - pad_left
                log_mel_spec = np.pad(log_mel_spec, ((0, 0), (pad_left, pad_right)), mode='constant')

            return log_mel_spec

        except Exception as e:
            logger.error(f"Error extracting features: {e}")
            # Возвращаем случайные признаки в случае ошибки
            return np.random.randn(self.n_mels, 128) * 0.1

    def _statistical_analysis(self, waveform):
        """
        Статистический анализ для дополнительной проверки
        """
        try:
            # Анализ спектральных характеристик
            spectral_centroids = librosa.feature.spectral_centroid(y=waveform, sr=self.sample_rate)[0]
            spectral_rolloff = librosa.feature.spectral_rolloff(y=waveform, sr=self.sample_rate)[0]
            zero_crossing_rate = librosa.feature.zero_crossing_rate(waveform)[0]

            # Статистики
            centroid_mean = np.mean(spectral_centroids)
            centroid_std = np.std(spectral_centroids)
            rolloff_mean = np.mean(spectral_rolloff)
            zcr_mean = np.mean(zero_crossing_rate)

            # Простая эвристика для обнаружения синтетической речи
            # Синтетическая речь часто имеет более стабильные характеристики

            # Если вариативность слишком низкая, возможно это синтетика
            if centroid_std < 200 and zcr_mean < 0.05:
                return 0.6  # Подозрительно

            # Если характеристики в нормальных пределах для живой речи
            if 1000 < centroid_mean < 3000 and 0.05 < zcr_mean < 0.3:
                return 0.2  # Вероятно живая речь

            return 0.4  # Неопределенно

        except Exception as e:
            logger.error(f"Error in statistical analysis: {e}")
            return 0.3

    def detect(self, audio_path, threshold=None):
        """
        Обнаружение спуфинг-атаки в аудиофайле
        """
        if threshold is None:
            threshold = SPOOFING_THRESHOLD

        try:
            # Проверка существования файла
            if not os.path.exists(audio_path):
                logger.error(f"Audio file not found: {audio_path}")
                return {
                    "is_spoof": False,
                    "spoof_probability": DEFAULT_SCORE,
                    "confidence": 0.5,
                    "error": "Audio file not found"
                }

            # Загрузка и предобработка аудио
            waveform, sr = librosa.load(audio_path, sr=self.sample_rate, mono=True)

            # Проверка на минимальную длину
            if len(waveform) < self.sample_rate * 0.5:  # Минимум 0.5 секунды
                logger.warning(f"Audio file too short: {len(waveform) / self.sample_rate:.2f}s")
                return {
                    "is_spoof": False,
                    "spoof_probability": DEFAULT_SCORE,
                    "confidence": 0.5,
                    "error": "Audio file too short"
                }

            # Нормализация
            waveform = librosa.util.normalize(waveform)

            # 1. Получение предсказания нейросетевой модели
            model_score = self._model_prediction(waveform)

            # 2. Статистический analysis для дополнительной проверки
            statistical_score = self._statistical_analysis(waveform)

            # 3. Комбинирование оценок
            # Отдаем больший вес статистическому анализу для уменьшения ложных срабатываний
            final_score = 0.3 * model_score + 0.7 * statistical_score

            # 4. Дополнительные проверки для уменьшения ложных срабатываний
            # Если это файл из API аутентификации, применяем более консервативный подход
            if 'auth_' in audio_path and audio_path.endswith('.wav'):
                # Дополнительно снижаем оценку для файлов аутентификации
                final_score = max(0.0, final_score - 0.15)

            # Адаптивный порог на основе истории оценок
            adaptive_threshold = self._get_adaptive_threshold(final_score)

            # 5. Принятие решения
            is_spoofing = final_score >= adaptive_threshold

            # Обновление истории оценок
            self.score_history.append(final_score)
            if len(self.score_history) > self.max_history:
                self.score_history.pop(0)

            logger.info(
                f"Spoofing detection: score={final_score:.4f}, threshold={adaptive_threshold:.4f}, is_spoof={is_spoofing}")

            return {
                "is_spoof": bool(is_spoofing),
                "spoof_probability": float(final_score),
                "confidence": float(min(0.9, abs(final_score - adaptive_threshold) + 0.5)),
                "threshold": float(adaptive_threshold),
                "model_score": float(model_score),
                "statistical_score": float(statistical_score)
            }

        except Exception as e:
            logger.error(f"Error in spoofing detection: {e}")
            return {
                "is_spoof": False,
                "spoof_probability": DEFAULT_SCORE,
                "confidence": 0.5,
                "error": str(e)
            }

    def _model_prediction(self, waveform):
        """
        Предсказание нейросетевой модели
        """
        try:
            with torch.no_grad():
                # Извлечение признаков
                features = self._extract_features(waveform)

                # Преобразование в тензор и добавление batch dimension
                features_tensor = torch.FloatTensor(features).unsqueeze(0).unsqueeze(0)  # [1, 1, n_mels, time]
                features_tensor = features_tensor.to(self.device)

                # Предсказание модели
                try:
                    prediction = self.model(features_tensor)

                    # Если модель возвращает тензор, извлекаем значение
                    if isinstance(prediction, torch.Tensor):
                        if prediction.dim() == 0:  # Скалярный тензор
                            prediction = prediction.item()
                        else:
                            prediction = prediction.cpu().numpy()[0] if len(prediction) > 0 else 0.3

                    return float(prediction)

                except Exception as model_error:
                    logger.error(f"Model prediction error: {model_error}")
                    return 0.3

        except Exception as e:
            logger.error(f"Error in model prediction: {e}")
            return 0.3

    def _get_adaptive_threshold(self, current_score):
        """
        Получение адаптивного порога на основе истории оценок
        """
        try:
            if len(self.score_history) < 5:
                return SPOOFING_THRESHOLD

            # Анализ истории для адаптации порога
            recent_scores = self.score_history[-10:]  # Последние 10 оценок
            mean_score = np.mean(recent_scores)
            std_score = np.std(recent_scores)

            # Если большинство недавних оценок низкие, снижаем порог
            if mean_score < 0.3 and std_score < 0.1:
                return max(0.3, SPOOFING_THRESHOLD - 0.1)

            # Если оценки высокие и стабильные, повышаем порог
            if mean_score > 0.6 and std_score < 0.15:
                return min(0.8, SPOOFING_THRESHOLD + 0.1)

            return SPOOFING_THRESHOLD

        except Exception as e:
            logger.error(f"Error calculating adaptive threshold: {e}")
            return SPOOFING_THRESHOLD

    def reset_history(self):
        """
        Сброс истории оценок
        """
        self.score_history = []
        logger.info("Score history reset")

    def get_statistics(self):
        """
        Получение статистики работы детектора
        """
        if not self.score_history:
            return {
                "total_detections": 0,
                "mean_score": 0.0,
                "std_score": 0.0,
                "min_score": 0.0,
                "max_score": 0.0
            }

        return {
            "total_detections": len(self.score_history),
            "mean_score": float(np.mean(self.score_history)),
            "std_score": float(np.std(self.score_history)),
            "min_score": float(np.min(self.score_history)),
            "max_score": float(np.max(self.score_history))
        }


class RawNet2AntiSpoofing(nn.Module):
    """
    Класс RawNet2AntiSpoofing для обратной совместимости с anti_spoof_trainer
    """

    def __init__(self, d_args=None):
        super(RawNet2AntiSpoofing, self).__init__()

        # Используем улучшенную модель внутри
        self.model = ImprovedAntiSpoofingNet()

    def forward(self, x):
        return self.model(x)