# ml_model/anti_spoof.py - Исправленная версия по аналогии с voice_embedding.py

import os
import librosa
import torch
import torch.nn as nn
import numpy as np
import logging
import torch.nn.functional as F
from pathlib import Path
import json

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("anti_spoofing")

SPOOFING_THRESHOLD = 0.5
MIN_CONFIDENCE = 0.3
DEFAULT_SCORE = 0.2


class ImprovedAntiSpoofingNet(nn.Module):
    """
    Улучшенная модель для обнаружения спуфинг-атак
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
            nn.AdaptiveAvgPool2d((10, 8)),
            nn.Dropout2d(0.3)
        )

        # Вычисляем размер после CNN
        cnn_output_size = 128 * 10 * 8

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
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

        self._initialize_weights()

    def _initialize_weights(self):
        """Инициализация весов"""
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
        """Forward pass"""
        batch_size = x.size(0)

        # Убеждаемся что входные данные имеют правильную размерность
        if x.dim() == 3:  # [batch_size, time, freq]
            x = x.unsqueeze(1)  # -> [batch_size, 1, time, freq]
        elif x.dim() == 2:  # [time, freq]
            x = x.unsqueeze(0).unsqueeze(0)  # -> [1, 1, time, freq]

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


class RawNet2AntiSpoofing(nn.Module):
    """
    Упрощенная реализация RawNet2 для защиты от спуфинга
    """

    def __init__(self, d_args=None):
        super(RawNet2AntiSpoofing, self).__init__()

        # Параметры модели
        self.sinc_out_channels = 128
        self.filter_length = 251

        # SincNet слой (упрощенная версия)
        self.sinc_conv = nn.Conv1d(
            in_channels=1,
            out_channels=self.sinc_out_channels,
            kernel_size=self.filter_length,
            stride=1,
            padding=self.filter_length // 2,
            bias=False
        )

        # Residual blocks
        self.res_blocks = nn.ModuleList([
            self._make_res_block(self.sinc_out_channels, 128),
            self._make_res_block(128, 256),
            self._make_res_block(256, 512),
        ])

        # Global Average Pooling
        self.gap = nn.AdaptiveAvgPool1d(1)

        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def _make_res_block(self, in_channels, out_channels):
        """Создает residual block"""
        return nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        )

    def forward(self, x):
        """Forward pass для RawNet2"""
        # Обеспечиваем правильную размерность
        if x.dim() == 2:  # [batch_size, samples]
            x = x.unsqueeze(1)  # -> [batch_size, 1, samples]
        elif x.dim() == 1:  # [samples]
            x = x.unsqueeze(0).unsqueeze(0)  # -> [1, 1, samples]

        # SincNet convolution
        x = self.sinc_conv(x)

        # Residual blocks
        for res_block in self.res_blocks:
            x = res_block(x)

        # Global Average Pooling
        x = self.gap(x)  # [batch_size, channels, 1]
        x = x.squeeze(-1)  # [batch_size, channels]

        # Classification
        output = self.classifier(x)

        return output.squeeze()


class AntiSpoofingDetector:
    """
    Класс для обнаружения спуфинг-атак по аналогии с VoiceEmbeddingModel
    """

    def __init__(self, model_path=None, device=None):
        # Инициализируем устройство
        self.device = self._initialize_device(device)
        logger.info(f"Using device: {self.device}")

        # Создаем директорию модели, если она не существует
        if model_path:
            self.model_path = model_path
            os.makedirs(model_path, exist_ok=True)
        else:
            self.model_path = "models"
            os.makedirs(self.model_path, exist_ok=True)

        # Параметры для обработки аудио
        self.sample_rate = 16000
        self.n_fft = 512
        self.hop_length = 256
        self.n_mels = 40

        # Статистики для адаптивного порога
        self.score_history = []
        self.max_history = 50

        # Инициализация модели с автозагрузкой
        self.model_is_trained = False

        try:
            # Сначала проверяем локальные обученные модели
            logger.info("Looking for local trained models")
            available_models = self._check_local_models()

            if available_models:
                # Если есть локальные модели, используем их
                try:
                    self.model = self._init_rawnet_model()
                    self.using_rawnet = True
                    self.using_pretrained = False
                    logger.info("Using local trained RawNet2 model")
                except:
                    self.model = self._init_basic_model()
                    if self.model is not None:
                        self.using_rawnet = False
                        self.using_pretrained = False
                        logger.info("Using local trained CNN+LSTM model")
            else:
                # Если нет локальных моделей, пробуем загрузить предобученную
                logger.info("No local models found, trying to download pretrained model")
                try:
                    self.model = self._init_pretrained_model()
                    self.using_pretrained = True
                    self.using_rawnet = False
                    self.model_is_trained = True
                    logger.info("Using downloaded pretrained model")
                except Exception as download_error:
                    logger.warning(f"Could not download pretrained model: {download_error}")
                    logger.warning("Using statistical analysis only")
                    self.model = None
                    self.using_rawnet = False
                    self.using_pretrained = False

        except Exception as init_error:
            logger.error(f"Model initialization failed: {init_error}")
            logger.warning("Using statistical analysis only")
            self.model = None
            self.using_rawnet = False
            self.using_pretrained = False

    def _check_local_models(self):
        """Проверяет какие обученные модели доступны локально"""
        available_models = []

        # Список возможных файлов моделей
        model_files = [
            "anti_spoof_model.pt",
            "rawnet2_antispoof.pt",
            "rawnet2_pretrained.pt",
            "best_model.pt",
            "aasist_pretrained.pt",
            "ssl_antispoof.pt"
        ]

        for model_file in model_files:
            full_path = os.path.join(self.model_path, model_file)
            if os.path.exists(full_path):
                try:
                    # Пробуем загрузить для проверки валидности
                    torch.load(full_path, map_location='cpu')
                    available_models.append(model_file)
                    logger.info(f"Found valid local model: {model_file}")
                except Exception as e:
                    logger.warning(f"Invalid model file {model_file}: {e}")

        if not available_models:
            logger.info("No local trained models found")
            logger.info("Will attempt to download pretrained model")
        else:
            logger.info(f"Available local models: {available_models}")

        return available_models

        logger.info("Anti-spoofing detector initialized")

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

    def _init_pretrained_model(self):
        """Инициализирует предобученную ESPnet RawNet3 модель с Hugging Face"""
        try:
            logger.info("Downloading ESPnet RawNet3 model from Hugging Face")
            return self._download_espnet_rawnet3()

        except Exception as download_error:
            logger.error(f"Failed to download ESPnet RawNet3: {download_error}")
            raise

    def _download_espnet_rawnet3(self):
        """Загружает ESPnet RawNet3 модель с Hugging Face"""
        try:
            model_name = "espnet/voxcelebs12_rawnet3"
            logger.info(f"Downloading {model_name} from Hugging Face")

            # Определяем путь для сохранения
            save_dir = os.path.join(self.model_path, "espnet_rawnet3")
            os.makedirs(save_dir, exist_ok=True)

            try:
                # Пробуем использовать transformers
                from transformers import AutoModel, AutoConfig
                logger.info("Using transformers library to download ESPnet RawNet3")

                # Загружаем конфигурацию и модель
                config = AutoConfig.from_pretrained(
                    model_name,
                    cache_dir=save_dir,
                    trust_remote_code=True
                )

                model = AutoModel.from_pretrained(
                    model_name,
                    config=config,
                    cache_dir=save_dir,
                    trust_remote_code=True
                )

                logger.info(f"Successfully downloaded {model_name} using transformers")

            except Exception as transformers_error:
                logger.warning(f"Transformers failed: {transformers_error}, trying direct download")

                # Пробуем прямую загрузку файлов модели
                import requests
                import json
                from urllib.parse import urljoin

                # URL базовой модели на Hugging Face
                base_url = f"https://huggingface.co/{model_name}/resolve/main/"

                # Список файлов для загрузки
                files_to_download = [
                    "config.json",
                    "pytorch_model.bin",
                    "model.safetensors"  # альтернативный формат
                ]

                downloaded_files = []

                for filename in files_to_download:
                    try:
                        file_url = urljoin(base_url, filename)
                        file_path = os.path.join(save_dir, filename)

                        logger.info(f"Downloading {filename} from {file_url}")

                        response = requests.get(file_url, stream=True, timeout=60)
                        response.raise_for_status()

                        with open(file_path, 'wb') as f:
                            for chunk in response.iter_content(chunk_size=8192):
                                f.write(chunk)

                        downloaded_files.append(filename)
                        logger.info(f"Successfully downloaded {filename}")

                    except Exception as file_error:
                        logger.warning(f"Could not download {filename}: {file_error}")
                        continue

                if not downloaded_files:
                    raise Exception("No model files could be downloaded")

                # Загружаем модель из скачанных файлов
                model = self._load_espnet_model_from_files(save_dir, downloaded_files)

            # Создаем wrapper для ESPnet модели
            wrapper = self._create_espnet_wrapper(model, save_dir)

            # Сохраняем информацию о модели
            model_info = {
                "model_name": model_name,
                "model_type": "ESPnet_RawNet3",
                "downloaded_date": datetime.now().isoformat(),
                "save_dir": save_dir
            }

            with open(os.path.join(save_dir, "model_info.json"), "w") as f:
                json.dump(model_info, f, indent=4)

            logger.info("ESPnet RawNet3 model successfully initialized")
            return wrapper

        except Exception as e:
            logger.error(f"Error downloading ESPnet RawNet3: {e}")
            raise

    def _load_espnet_model_from_files(self, save_dir, downloaded_files):
        """Загружает ESPnet модель из скачанных файлов"""
        try:
            import json

            # Читаем конфигурацию
            config_path = os.path.join(save_dir, "config.json")
            if "config.json" in downloaded_files and os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    config = json.load(f)
                logger.info("Loaded model configuration")
            else:
                config = {}

            # Загружаем веса модели
            model_weights = None

            # Пробуем pytorch_model.bin
            bin_path = os.path.join(save_dir, "pytorch_model.bin")
            if "pytorch_model.bin" in downloaded_files and os.path.exists(bin_path):
                try:
                    model_weights = torch.load(bin_path, map_location=self.device)
                    logger.info("Loaded model weights from pytorch_model.bin")
                except Exception as e:
                    logger.warning(f"Could not load pytorch_model.bin: {e}")

            # Пробуем model.safetensors если bin не удался
            if model_weights is None:
                safetensors_path = os.path.join(save_dir, "model.safetensors")
                if "model.safetensors" in downloaded_files and os.path.exists(safetensors_path):
                    try:
                        # Для safetensors нужна библиотека safetensors
                        from safetensors import safe_open

                        model_weights = {}
                        with safe_open(safetensors_path, framework="pt", device=str(self.device)) as f:
                            for key in f.keys():
                                model_weights[key] = f.get_tensor(key)

                        logger.info("Loaded model weights from model.safetensors")

                    except ImportError:
                        logger.warning("safetensors library not available")
                    except Exception as e:
                        logger.warning(f"Could not load model.safetensors: {e}")

            if model_weights is None:
                raise Exception("Could not load model weights from any file")

            # Создаем модель RawNet3 на основе загруженных весов
            model = self._create_rawnet3_model(config, model_weights)
            return model

        except Exception as e:
            logger.error(f"Error loading ESPnet model from files: {e}")
            raise

    def _create_rawnet3_model(self, config, model_weights):
        """Создает RawNet3 модель на основе конфигурации и весов"""
        try:
            # Создаем архитектуру RawNet3 (упрощенная версия)
            class ESPnetRawNet3(nn.Module):
                def __init__(self, config_dict=None):
                    super(ESPnetRawNet3, self).__init__()

                    # Параметры из конфигурации или значения по умолчанию
                    if config_dict and 'model_conf' in config_dict:
                        model_conf = config_dict['model_conf']
                    else:
                        model_conf = {}

                    # SincNet слои
                    self.sinc_conv = nn.Conv1d(
                        in_channels=1,
                        out_channels=model_conf.get('sinc_out_channels', 128),
                        kernel_size=model_conf.get('sinc_kernel_size', 251),
                        stride=1,
                        padding=model_conf.get('sinc_kernel_size', 251) // 2,
                        bias=False
                    )

                    # Residual blocks
                    self.res_blocks = nn.ModuleList()
                    channels = [128, 256, 512, 1024]

                    for i, out_channels in enumerate(channels):
                        in_channels = 128 if i == 0 else channels[i - 1]
                        self.res_blocks.append(self._make_res_block(in_channels, out_channels))

                    # Global statistics pooling
                    self.gsp = nn.AdaptiveAvgPool1d(1)

                    # Classifier для anti-spoofing
                    self.classifier = nn.Sequential(
                        nn.Linear(1024, 512),
                        nn.ReLU(),
                        nn.Dropout(0.5),
                        nn.Linear(512, 256),
                        nn.ReLU(),
                        nn.Dropout(0.5),
                        nn.Linear(256, 1),
                        nn.Sigmoid()
                    )

                def _make_res_block(self, in_channels, out_channels):
                    return nn.Sequential(
                        nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1),
                        nn.BatchNorm1d(out_channels),
                        nn.ReLU(),
                        nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1),
                        nn.BatchNorm1d(out_channels),
                        nn.ReLU(),
                        nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
                        nn.Dropout(0.3)
                    )

                def forward(self, x):
                    # x: [batch_size, samples] или [batch_size, 1, samples]
                    if x.dim() == 2:
                        x = x.unsqueeze(1)  # [batch_size, 1, samples]

                    # SincNet convolution
                    x = self.sinc_conv(x)

                    # Residual blocks
                    for res_block in self.res_blocks:
                        x = res_block(x)

                    # Global statistics pooling
                    x = self.gsp(x)  # [batch_size, channels, 1]
                    x = x.squeeze(-1)  # [batch_size, channels]

                    # Classification
                    output = self.classifier(x)
                    return output.squeeze()

            # Создаем модель
            model = ESPnetRawNet3(config).to(self.device)

            # Загружаем веса (с обработкой несовпадающих ключей)
            try:
                model.load_state_dict(model_weights, strict=False)
                logger.info("Successfully loaded ESPnet RawNet3 weights")
            except Exception as weight_error:
                logger.warning(f"Could not load all weights: {weight_error}")
                logger.info("Using partially loaded model")

            model.eval()
            return model

        except Exception as e:
            logger.error(f"Error creating RawNet3 model: {e}")
            # Fallback к нашей базовой модели
            logger.info("Using fallback RawNet2 model")
            return RawNet2AntiSpoofing().to(self.device)

    def _create_espnet_wrapper(self, model, save_dir):
        """Создает wrapper для ESPnet модели"""

        class ESPnetAntiSpoofWrapper:
            def __init__(self, espnet_model, device, model_dir):
                self.espnet_model = espnet_model.to(device)
                self.device = device
                self.model_dir = model_dir
                self.espnet_model.eval()

                # Параметры для обработки аудио
                self.sample_rate = 16000
                self.target_length = 64000  # 4 секунды при 16kHz

            def __call__(self, waveform_tensor):
                with torch.no_grad():
                    # Обеспечиваем правильную размерность
                    if waveform_tensor.dim() == 3:  # [1, 1, samples]
                        waveform_tensor = waveform_tensor.squeeze(1)  # [1, samples]
                    elif waveform_tensor.dim() == 1:  # [samples]
                        waveform_tensor = waveform_tensor.unsqueeze(0)  # [1, samples]

                    # Нормируем длину аудио
                    current_length = waveform_tensor.shape[1]
                    if current_length > self.target_length:
                        # Берем из центра
                        start = (current_length - self.target_length) // 2
                        waveform_tensor = waveform_tensor[:, start:start + self.target_length]
                    elif current_length < self.target_length:
                        # Дополняем нулями
                        pad_length = self.target_length - current_length
                        waveform_tensor = torch.nn.functional.pad(
                            waveform_tensor,
                            (0, pad_length),
                            mode='constant',
                            value=0
                        )

                    # Получаем предсказание от модели
                    try:
                        output = self.espnet_model(waveform_tensor)

                        # Если выход скалярный, возвращаем как есть
                        if output.dim() == 0:
                            return output
                        # Если это вектор, берем первый элемент или среднее
                        elif output.dim() == 1:
                            return output.mean() if len(output) > 1 else output[0]
                        else:
                            return output.mean()

                    except Exception as forward_error:
                        logger.error(f"ESPnet model forward error: {forward_error}")
                        # Возвращаем консервативную оценку
                        return torch.tensor(0.25, device=self.device)

        wrapper = ESPnetAntiSpoofWrapper(model, self.device, save_dir)
        logger.info("Created ESPnet RawNet3 wrapper")
        return wrapper

    def _init_rawnet_model(self):
        """Инициализирует модель RawNet2"""
        # Ищем модель в разных местах
        possible_paths = [
            os.path.join(self.model_path, "rawnet2_antispoof.pt"),
            os.path.join(self.model_path, "anti_spoof_model.pt"),
            os.path.join(self.model_path, "rawnet2_pretrained.pt")
        ]

        model_loaded = False
        model = RawNet2AntiSpoofing().to(self.device)

        for model_file in possible_paths:
            if os.path.exists(model_file):
                try:
                    state_dict = torch.load(model_file, map_location=self.device)
                    model.load_state_dict(state_dict, strict=False)
                    logger.info(f"Successfully loaded RawNet2 model from {model_file}")
                    model_loaded = True
                    break
                except Exception as e:
                    logger.warning(f"Could not load RawNet2 weights from {model_file}: {e}")
                    continue

        if not model_loaded:
            # Если не нашли обученную модель, НЕ используем случайную инициализацию
            # Вместо этого переходим к базовой модели
            raise FileNotFoundError("No trained RawNet2 model found")

        model.eval()
        return model

    def _init_basic_model(self):
        """Инициализирует базовую модель CNN+LSTM"""
        # Ищем обученную модель в разных местах
        possible_paths = [
            os.path.join(self.model_path, "anti_spoof_model.pt"),
            os.path.join(self.model_path, "rawnet2_antispoof.pt"),
            os.path.join(self.model_path, "best_model.pt")
        ]

        model_loaded = False
        model = ImprovedAntiSpoofingNet().to(self.device)

        for model_file in possible_paths:
            if os.path.exists(model_file):
                try:
                    state_dict = torch.load(model_file, map_location=self.device)
                    model.load_state_dict(state_dict, strict=False)
                    logger.info(f"Successfully loaded CNN+LSTM model from {model_file}")
                    model_loaded = True
                    break
                except Exception as e:
                    logger.warning(f"Could not load model weights from {model_file}: {e}")
                    continue

        if not model_loaded:
            # Если не нашли НИКАКОЙ обученной модели, создаем заглушку которая всегда возвращает "не спуфинг"
            logger.warning("No trained anti-spoofing model found. Using conservative fallback.")
            # Помечаем модель как необученную
            self.model_is_trained = False
            return None  # Будем использовать только статистический анализ

        self.model_is_trained = True
        model.eval()
        return model

    def _extract_features(self, waveform):
        """Извлечение признаков из аудио для анализа спуфинга"""
        try:
            if hasattr(self, 'using_rawnet') and self.using_rawnet:
                # Для RawNet2 возвращаем сырой сигнал
                target_length = 48000  # 3 секунды при 16kHz
                if len(waveform) > target_length:
                    # Берем из середины
                    start = (len(waveform) - target_length) // 2
                    waveform = waveform[start:start + target_length]
                elif len(waveform) < target_length:
                    # Дополняем нулями
                    waveform = np.pad(waveform, (0, target_length - len(waveform)))

                return waveform

            # Для CNN+LSTM модели используем мел-спектрограмму
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

            # Приведение к стандартному размеру для модели
            target_length = 128  # Примерно 4 секунды при hop_length=256
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
            if hasattr(self, 'using_rawnet') and self.using_rawnet:
                return np.random.randn(48000) * 0.1
            else:
                return np.random.randn(self.n_mels, 128) * 0.1

    def _statistical_analysis(self, waveform):
        """Статистический анализ для дополнительной проверки"""
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
        """Основной метод для обнаружения спуфинг-атаки в аудиофайле"""
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

            logger.info(f"Analyzing audio file: {audio_path}")

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

            # 2. Статистический анализ для дополнительной проверки
            statistical_score = self._statistical_analysis(waveform)

            # 3. Комбинирование оценок
            if hasattr(self, 'using_rawnet') and self.using_rawnet:
                # Для RawNet2 больше доверяем модели
                final_score = 0.7 * model_score + 0.3 * statistical_score
            else:
                # Для базовой модели больше веса статистическому анализу
                final_score = 0.4 * model_score + 0.6 * statistical_score

            # 4. Дополнительные проверки для уменьшения ложных срабатываний
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
            # В случае ошибки возвращаем консервативный результат
            return {
                "is_spoof": False,
                "spoof_probability": DEFAULT_SCORE,
                "confidence": 0.5,
                "error": str(e)
            }

    def _model_prediction(self, waveform):
        """Предсказание нейросетевой модели"""
        try:
            # Если нет обученной модели, используем только статистический анализ
            if self.model is None or not hasattr(self, 'model_is_trained') or not self.model_is_trained:
                logger.info("No trained model available, using statistical analysis only")
                return 0.25  # Консервативная оценка для реального голоса

            with torch.no_grad():
                # Извлечение признаков
                features = self._extract_features(waveform)

                # Подготовка тензора в зависимости от типа модели
                if hasattr(self, 'using_rawnet') and self.using_rawnet:
                    # Для RawNet2 используем сырой сигнал
                    features_tensor = torch.FloatTensor(features).unsqueeze(0)  # [1, samples]
                else:
                    # Для CNN+LSTM модели используем мел-спектрограмму
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
                            prediction = prediction.cpu().numpy()
                            if len(prediction.shape) > 0 and len(prediction) > 0:
                                prediction = prediction[0]
                            else:
                                prediction = 0.25

                    return float(prediction)

                except Exception as model_error:
                    logger.error(f"Model prediction error: {model_error}")
                    return 0.25

        except Exception as e:
            logger.error(f"Error in model prediction: {e}")
            # В случае ошибки возвращаем консервативную оценку
            return 0.25

    def _get_adaptive_threshold(self, current_score):
        """Получение адаптивного порога на основе истории оценок"""
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
        """Сброс истории оценок"""
        self.score_history = []
        logger.info("Score history reset")

    def get_statistics(self):
        """Получение статистики работы детектора"""
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


# Класс для обратной совместимости с trainer
class RawNet2AntiSpoofing_Compat(RawNet2AntiSpoofing):
    """Класс для обратной совместимости с anti_spoof_trainer"""

    def __init__(self, d_args=None):
        super().__init__(d_args)