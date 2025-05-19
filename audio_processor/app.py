from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import os
import tempfile
import shutil
import numpy as np
import librosa
import soundfile as sf
import logging
import queue
from pathlib import Path
import json
import threading
import time
import requests
from contextlib import contextmanager

# Настройка логирования
log_path = "/app/logs"
if not os.path.exists(log_path):
    try:
        os.makedirs(log_path, exist_ok=True)
        log_file = os.path.join(log_path, "audio_processor.log")
    except (OSError, PermissionError):
        # Если невозможно создать директорию, используем временную директорию
        log_path = tempfile.gettempdir()
        log_file = os.path.join(log_path, "audio_processor.log")
else:
    log_file = os.path.join(log_path, "audio_processor.log")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(log_file),
    ]
)
logger = logging.getLogger("audio_processor")


# Добавьте в верхнюю часть файла, после импортов:
CONTAINER_MODE = os.environ.get("CONTAINER_MODE", "False").lower() in ("true", "1", "yes")
logger.info(f"Container mode: {CONTAINER_MODE}")
# Проверяем наличие зависимостей для работы с аудио устройствами
# и импортируем их только если они доступны
AUDIO_HARDWARE_AVAILABLE = True
try:
    import sounddevice as sd
    import pyaudio
except ImportError:
    AUDIO_HARDWARE_AVAILABLE = False
    logging.warning("Audio hardware libraries not available. Recording functionality will be limited.")

app = FastAPI(title="Сервис обработки аудио", version="1.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Конфигурация
AUDIO_PATH = os.environ.get("AUDIO_PATH", "/shared/audio")
TEMP_AUDIO_PATH = os.path.join(AUDIO_PATH, "temp")
CONTAINER_MODE = os.environ.get("CONTAINER_MODE", "False").lower() in ("true", "1", "yes")

# Создание директорий, если не существуют
try:
    os.makedirs(AUDIO_PATH, exist_ok=True)
    os.makedirs(TEMP_AUDIO_PATH, exist_ok=True)
except (OSError, PermissionError) as e:
    logger.warning(f"Cannot create audio directories: {e}. Using temporary directory.")
    AUDIO_PATH = os.path.join(tempfile.gettempdir(), "audio")
    TEMP_AUDIO_PATH = os.path.join(AUDIO_PATH, "temp")
    os.makedirs(AUDIO_PATH, exist_ok=True)
    os.makedirs(TEMP_AUDIO_PATH, exist_ok=True)

# Глобальные переменные для записи аудио
recording = False
kpp_mode_active = False
kpp_thread = None


# Модели данных
class AudioFile(BaseModel):
    file_path: str


class AudioRecordingOptions(BaseModel):
    duration: int = 5  # длительность записи в секундах
    sample_rate: int = 16000
    channels: int = 1
    device_index: Optional[int] = None


class AudioProcessingResult(BaseModel):
    success: bool
    processed_path: Optional[str] = None
    sample_rate: int
    duration: float
    channels: int
    message: str = "Audio processed successfully"


class KppModeOptions(BaseModel):
    enabled: bool
    sensitivity: float = 0.03  # чувствительность определения речи
    min_speech_duration: float = 1.0  # минимальная длительность речи в секундах
    auto_reset_delay: int = 5  # задержка до сброса в секундах
    callback_url: Optional[str] = None  # URL для отправки результатов


class AudioDeviceInfo(BaseModel):
    index: int
    name: str
    max_input_channels: int
    max_output_channels: int
    default_sample_rate: float
    is_input: bool
    is_output: bool


class AudioBuffer:
    def __init__(self, max_size=100):
        self.buffer = queue.Queue(maxsize=max_size)

    def callback(self, indata, frames, time, status):
        """Callback для потока аудио"""
        if status:
            logger.warning(f"Audio callback status: {status}")
        try:
            self.buffer.put(indata.copy())
        except queue.Full:
            logger.warning("Audio buffer overflow")

    def get_audio(self, timeout=0.1):
        """Получение данных из буфера"""
        try:
            return self.buffer.get(timeout=timeout)
        except queue.Empty:
            return None

# Вспомогательные функции
@contextmanager
def safe_audio_operation():
    """Контекстный менеджер для безопасных операций с аудио"""
    try:
        yield
    except Exception as e:
        logger.error(f"Audio operation error: {str(e)}")
        if "No such file or directory" in str(e):
            raise HTTPException(status_code=404, detail="Audio file not found")
        elif "sample rate" in str(e).lower() or "channels" in str(e).lower():
            raise HTTPException(status_code=400, detail="Invalid audio format")
        elif "permission" in str(e).lower():
            raise HTTPException(status_code=403, detail="Permission denied")
        else:
            raise HTTPException(status_code=500, detail=str(e))


def send_callback(url: str, data: Dict[str, Any]):
    """Отправка данных на callback URL"""
    try:
        response = requests.post(url, json=data, timeout=5)
        return response.status_code == 200
    except Exception as e:
        logger.error(f"Error sending callback: {e}")
        return False


# Эндпоинты API
@app.get("/health")
async def health_check():
    """Проверка работоспособности сервиса"""
    return {
        "status": "ok",
        "version": "1.0.0",
        "container_mode": CONTAINER_MODE,
        "audio_hardware_available": AUDIO_HARDWARE_AVAILABLE,
        "storage_path": AUDIO_PATH
    }


@app.get("/audio_devices", response_model=List[AudioDeviceInfo])
async def get_audio_devices():
    """Получение списка доступных аудиоустройств"""
    if not AUDIO_HARDWARE_AVAILABLE:
        return []

    try:
        # Используем PyAudio для получения списка устройств
        p = pyaudio.PyAudio()
        devices = []

        for i in range(p.get_device_count()):
            device_info = p.get_device_info_by_index(i)
            devices.append(AudioDeviceInfo(
                index=device_info['index'],
                name=device_info['name'],
                max_input_channels=device_info['maxInputChannels'],
                max_output_channels=device_info['maxOutputChannels'],
                default_sample_rate=device_info['defaultSampleRate'],
                is_input=device_info['maxInputChannels'] > 0,
                is_output=device_info['maxOutputChannels'] > 0
            ))

        p.terminate()
        return devices
    except Exception as e:
        logger.error(f"Error getting audio devices: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get audio devices: {str(e)}")


@app.post("/record_audio", response_model=AudioProcessingResult)
async def record_audio(options: AudioRecordingOptions, background_tasks: BackgroundTasks):
    """Запись аудио с микрофона"""
    global recording

    # Проверка доступности аудио оборудования
    if not AUDIO_HARDWARE_AVAILABLE or CONTAINER_MODE:
        # Если недоступно, возвращаем симуляцию
        logger.info("Recording not available in container mode or without audio hardware")
        return AudioProcessingResult(
            success=True,
            processed_path=None,
            sample_rate=options.sample_rate,
            duration=options.duration,
            channels=options.channels,
            message="Audio recording simulated - hardware not available"
        )

    if recording:
        raise HTTPException(status_code=400, detail="Already recording")

    try:
        recording = True

        # Генерация имени файла
        timestamp = int(time.time())
        output_filename = f"recording_{timestamp}.wav"
        output_path = os.path.join(TEMP_AUDIO_PATH, output_filename)

        # Проверка наличия аудиоустройств
        devices = sd.query_devices()
        if options.device_index is not None and options.device_index >= len(devices):
            recording = False
            raise HTTPException(status_code=400, detail=f"Device index {options.device_index} out of range")

        # Запись аудио с обработкой ошибок
        try:
            audio_data = sd.rec(
                int(options.duration * options.sample_rate),
                samplerate=options.sample_rate,
                channels=options.channels,
                dtype='float32',
                device=options.device_index
            )
            sd.wait()  # Ждем окончания записи
        except sd.PortAudioError as e:
            recording = False
            logger.error(f"PortAudio error during recording: {e}")
            raise HTTPException(status_code=500, detail=f"Audio recording failed: {str(e)}")

        # Проверка данных
        if audio_data.size == 0 or np.max(np.abs(audio_data)) < 0.001:
            logger.warning("Recorded audio seems to be silent or empty")

        # Нормализация
        audio_data = librosa.util.normalize(audio_data)

        # Сохранение файла
        with safe_audio_operation():
            sf.write(output_path, audio_data, options.sample_rate)

        # Обработка записанного аудио
        processed_path, audio_info = preprocess_audio(output_path)

        recording = False

        return AudioProcessingResult(
            success=True,
            processed_path=processed_path,
            sample_rate=audio_info["sample_rate"],
            duration=audio_info["duration"],
            channels=audio_info["channels"],
            message="Audio recorded and processed successfully"
        )
    except Exception as e:
        recording = False
        logger.error(f"Error recording audio: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/process_audio", response_model=AudioProcessingResult)
async def process_audio(file: UploadFile = File(...)):
    """Обработка загруженного аудиофайла"""
    try:
        # Проверка расширения файла
        if not file.filename or '.' not in file.filename:
            raise HTTPException(status_code=400, detail="Invalid filename")

        extension = file.filename.split('.')[-1].lower()
        valid_extensions = ['wav', 'mp3', 'ogg', 'flac', 'm4a']

        if extension not in valid_extensions:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file format. Supported formats: {', '.join(valid_extensions)}"
            )

        # Создание временного файла для сохранения загруженного аудио
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{extension}") as temp_file:
            shutil.copyfileobj(file.file, temp_file)
            temp_path = temp_file.name

        # Обработка аудиофайла
        with safe_audio_operation():
            processed_path, audio_info = preprocess_audio(temp_path)

        return AudioProcessingResult(
            success=True,
            processed_path=processed_path,
            sample_rate=audio_info["sample_rate"],
            duration=audio_info["duration"],
            channels=audio_info["channels"]
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing audio: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Удаление временного файла
        if 'temp_path' in locals() and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except OSError as e:
                logger.warning(f"Failed to remove temporary file: {e}")


@app.post("/enhance_audio", response_model=AudioProcessingResult)
async def enhance_audio(audio_file: AudioFile):
    """Улучшение качества аудиофайла"""
    try:
        # Проверка существования файла
        if not os.path.exists(audio_file.file_path):
            raise HTTPException(status_code=404, detail="Audio file not found")

        # Улучшение качества аудиофайла
        with safe_audio_operation():
            processed_path, audio_info = enhance_audio_quality(audio_file.file_path)

        return AudioProcessingResult(
            success=True,
            processed_path=processed_path,
            sample_rate=audio_info["sample_rate"],
            duration=audio_info["duration"],
            channels=audio_info["channels"],
            message="Audio enhanced successfully"
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error enhancing audio: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/kpp_mode")
async def toggle_kpp_mode(options: KppModeOptions):
    """Включение/выключение режима КПП (непрерывное прослушивание)"""
    global kpp_mode_active, kpp_thread

    # Проверка доступности аудио оборудования
    if not AUDIO_HARDWARE_AVAILABLE or CONTAINER_MODE:
        return {
            "status": "unavailable",
            "message": "KPP mode not available in container mode or without audio hardware"
        }

    try:
        if options.enabled and not kpp_mode_active:
            # Запуск режима КПП
            kpp_mode_active = True
            kpp_thread = threading.Thread(target=kpp_mode_worker, args=(options,))
            kpp_thread.daemon = True
            kpp_thread.start()
            return {"status": "started", "message": "KPP mode activated"}
        elif not options.enabled and kpp_mode_active:
            # Остановка режима КПП
            kpp_mode_active = False
            if kpp_thread:
                kpp_thread.join(timeout=1.0)
            return {"status": "stopped", "message": "KPP mode deactivated"}
        elif options.enabled and kpp_mode_active:
            return {"status": "already_running", "message": "KPP mode is already active"}
        else:
            return {"status": "already_stopped", "message": "KPP mode is already inactive"}
    except Exception as e:
        logger.error(f"Error toggling KPP mode: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Функции обработки аудио
def preprocess_audio(file_path, target_sr=16000, target_channels=1):
    """
    Предобработка аудиофайла - ресемплирование, конвертация в моно, нормализация.

    Args:
        file_path: Путь к аудиофайлу
        target_sr: Целевая частота дискретизации
        target_channels: Целевое количество каналов (1 = моно)

    Returns:
        tuple: (путь к обработанному файлу, информация об аудио)
    """
    try:
        # Проверка существования файла
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        # Загрузка аудио с обработкой ошибок
        try:
            y, sr = librosa.load(file_path, sr=target_sr, mono=target_channels == 1)
        except Exception as e:
            logger.error(f"Error loading audio file: {e}")
            # Попытка альтернативной загрузки через soundfile
            try:
                y, sr = sf.read(file_path)
                # Конвертация в моно если необходимо
                if y.ndim > 1 and target_channels == 1:
                    y = np.mean(y, axis=1)
                # Ресемплирование если необходимо
                if sr != target_sr:
                    y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
                    sr = target_sr
            except Exception as sf_error:
                logger.error(f"Alternative loading failed: {sf_error}")
                raise e

        # Проверка аудио данных
        if len(y) == 0:
            raise ValueError("Empty audio data")

        # Нормализация амплитуды
        y = librosa.util.normalize(y)

        # Удаление тишины
        y, _ = librosa.effects.trim(y, top_db=20)

        # Получение информации об аудио
        duration = librosa.get_duration(y=y, sr=sr)

        # Генерация имени выходного файла
        output_filename = f"processed_{Path(file_path).stem}_{int(duration)}s.wav"
        output_path = os.path.join(AUDIO_PATH, output_filename)

        # Сохранение обработанного аудио
        sf.write(output_path, y, sr)

        audio_info = {
            "sample_rate": sr,
            "duration": duration,
            "channels": 1 if target_channels == 1 else len(y)
        }

        logger.info(f"Processed audio saved to {output_path}")

        return output_path, audio_info
    except Exception as e:
        logger.error(f"Error in preprocess_audio: {e}")
        raise


def enhance_audio_quality(file_path, target_sr=16000):
    """
    Улучшение качества аудио - шумоподавление, эквализация.

    Args:
        file_path: Путь к аудиофайлу
        target_sr: Целевая частота дискретизации

    Returns:
        tuple: (путь к улучшенному файлу, информация об аудио)
    """
    try:
        # Проверка существования файла
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        # Загрузка аудио
        y, sr = librosa.load(file_path, sr=target_sr)

        # Проверка аудио данных
        if len(y) == 0:
            raise ValueError("Empty audio data")

        # Нормализация
        y = librosa.util.normalize(y)

        # Шумоподавление (простая реализация)
        # Вычисление спектра шума из начала записи (предполагаем, что там тишина)
        n_fft = 2048

        # Проверка длины аудио для выборки шума
        silence_duration = min(0.1, len(y) / sr / 2)  # Не более половины аудио
        noise_sample = y[:int(sr * silence_duration)]

        if len(noise_sample) == 0:
            # Если не хватает данных, используем первую часть аудио
            noise_sample = y[:min(int(sr * 0.1), len(y))]

        # Проверка размера выборки шума
        if len(noise_sample) < n_fft // 2:
            n_fft = max(256, 2 ** int(np.log2(len(noise_sample) * 2)))

        noise_stft = librosa.stft(noise_sample, n_fft=n_fft)
        noise_power = np.mean(np.abs(noise_stft) ** 2, axis=1)

        # Применение спектрального вычитания
        y_stft = librosa.stft(y, n_fft=n_fft)
        y_power = np.abs(y_stft) ** 2
        y_phase = np.angle(y_stft)

        # Вычитание шума с порогом
        noise_mult = 2.0  # Множитель для шумового профиля
        gain = np.maximum(y_power - noise_mult * noise_power.reshape(-1, 1), 0) / (y_power + 1e-8)
        y_stft_denoised = gain * np.exp(1j * y_phase) * np.abs(y_stft)

        # Обратное преобразование
        y_denoised = librosa.istft(y_stft_denoised, length=len(y))

        # Финальная нормализация
        y_denoised = librosa.util.normalize(y_denoised)

        # Получение информации об аудио
        duration = librosa.get_duration(y=y_denoised, sr=sr)

        # Генерация имени выходного файла
        output_filename = f"enhanced_{Path(file_path).stem}.wav"
        output_path = os.path.join(AUDIO_PATH, output_filename)

        # Сохранение улучшенного аудио
        sf.write(output_path, y_denoised, sr)

        audio_info = {
            "sample_rate": sr,
            "duration": duration,
            "channels": 1
        }

        logger.info(f"Enhanced audio saved to {output_path}")

        return output_path, audio_info
    except Exception as e:
        logger.error(f"Error in enhance_audio_quality: {e}")
        raise


def detect_speech_activity(audio_data, sample_rate, sensitivity=0.03):
    """
    Улучшенное обнаружение речи с использованием спектральных характеристик.

    Args:
        audio_data: Аудиоданные
        sample_rate: Частота дискретизации
        sensitivity: Порог чувствительности

    Returns:
        bool: True если обнаружена речь, иначе False
    """
    try:
        # Преобразование в монофонический массив (если нужно)
        if len(audio_data.shape) > 1:
            audio_data = np.mean(audio_data, axis=1)

        # 1. Проверка на энергию сигнала
        energy = np.mean(np.abs(audio_data))
        energy_detected = energy > sensitivity

        if not energy_detected:
            return False

        # 2. Спектральный анализ (STFT)
        n_fft = min(512, len(audio_data))
        if n_fft < 64:  # Если данных слишком мало
            return energy_detected

        hop_length = n_fft // 4
        stft = librosa.stft(y=audio_data, n_fft=n_fft, hop_length=hop_length)
        spectrogram = np.abs(stft)

        # 3. Анализ диапазона частот человеческой речи (300-3400 Гц)
        freq_bins = librosa.fft_frequencies(sr=sample_rate, n_fft=n_fft)
        speech_range = (300, 3400)

        speech_mask = (freq_bins >= speech_range[0]) & (freq_bins <= speech_range[1])
        speech_energy = np.mean(spectrogram[speech_mask, :])

        total_energy = np.mean(spectrogram)
        speech_ratio = speech_energy / (total_energy + 1e-10)  # Избегаем деления на 0

        # 4. Проверка соотношения энергии в диапазоне речи к общей энергии
        is_speech = speech_ratio > 0.5 and energy_detected

        return is_speech
    except Exception as e:
        logger.error(f"Error in advanced speech detection: {e}")
        # При ошибке возвращаем результат обычной проверки энергии
        return energy > sensitivity

# В функции kpp_mode_worker:
def kpp_mode_worker(options: KppModeOptions):
    """
    Рабочий поток для режима КПП (непрерывное прослушивание).

    Args:
        options: Параметры режима КПП
    """
    global kpp_mode_active

    logger.info("Starting KPP mode worker")

    # Параметры записи
    sample_rate = 16000
    block_size = 4096  # Увеличиваем для лучшей производительности
    channels = 1

    # Буфер для записи
    audio_buffer = []

    # Состояние записи
    is_recording = False
    speech_start_time = 0

    try:
        # Инициализация аудиопотока
        try:
            logger.info("Opening audio stream")
            # Получаем список устройств для логирования
            devices_info = sd.query_devices()
            logger.info(f"Available audio devices: {len(devices_info)}")
            for i, device in enumerate(devices_info):
                logger.info(
                    f"Device {i}: {device['name']} - In: {device['max_input_channels']}, Out: {device['max_output_channels']}")

            # Выбираем устройство с максимальным числом входных каналов
            input_devices = [(i, device) for i, device in enumerate(devices_info) if device['max_input_channels'] > 0]

            if not input_devices:
                logger.error("No input devices found")
                raise RuntimeError("No input devices found")

            # Выбираем первое устройство ввода
            device_index = input_devices[0][0]
            logger.info(f"Using device {device_index}: {devices_info[device_index]['name']}")

            # Создаем поток с выбранным устройством
            stream = sd.InputStream(
                samplerate=sample_rate,
                blocksize=block_size,
                channels=channels,
                dtype='float32',
                device=device_index
            )
            stream.start()
            logger.info("Audio stream started successfully")
        except Exception as e:
            logger.error(f"Error starting audio stream: {str(e)}")
            kpp_mode_active = False
            return

        logger.info("KPP mode worker running in real-time mode")

        # Параметры обнаружения речи
        # (может потребоваться настройка в зависимости от шума окружающей среды)
        energy_threshold = options.sensitivity
        speech_timeout = 1.5  # Время в секундах тишины, после которого считаем речь завершенной
        min_speech_duration = options.min_speech_duration

        # Таймеры и счетчики
        silence_start = None
        recording_started = None

        while kpp_mode_active:
            # Чтение блока аудио
            audio_block, overflowed = stream.read(block_size)

            if overflowed:
                logger.warning("Audio input buffer overflowed")

            # Вычисление энергии блока
            energy = np.mean(np.abs(audio_block))
            is_speech = energy > energy_threshold

            # Логика обнаружения речи
            if is_speech:
                # Если это начало речи
                if not is_recording:
                    logger.info(f"Speech detected (energy: {energy:.6f}, threshold: {energy_threshold:.6f})")
                    is_recording = True
                    recording_started = time.time()
                    silence_start = None
                    audio_buffer = [audio_block]
                else:
                    # Продолжаем запись речи
                    audio_buffer.append(audio_block)
                    silence_start = None
            else:
                # Если нет речи, но запись активна - возможно, это пауза
                if is_recording:
                    # Если это начало тишины
                    if silence_start is None:
                        silence_start = time.time()

                    # Если тишина слишком долгая, завершаем запись
                    if time.time() - silence_start > speech_timeout:
                        speech_duration = time.time() - recording_started

                        if speech_duration >= min_speech_duration:
                            # Речь достаточно длинная, обрабатываем
                            logger.info(f"Speech ended, duration: {speech_duration:.2f}s")

                            # Объединение блоков в один аудиофайл
                            audio_data = np.concatenate(audio_buffer)

                            # Сохранение временного файла
                            timestamp = int(time.time())
                            temp_file = os.path.join(TEMP_AUDIO_PATH, f"kpp_recording_{timestamp}.wav")
                            sf.write(temp_file, audio_data, sample_rate)

                            # Обработка аудио
                            try:
                                processed_path, audio_info = preprocess_audio(temp_file)
                                logger.info(f"KPP recording processed: {processed_path}")

                                # Если указан callback_url, отправляем результат
                                if options.callback_url:
                                    try:
                                        import requests

                                        logger.info(f"Sending result to callback URL: {options.callback_url}")
                                        response = requests.get(
                                            options.callback_url,
                                            params={"file_path": processed_path}
                                        )
                                        logger.info(f"Callback response: {response.status_code}")
                                    except Exception as e:
                                        logger.error(f"Error sending callback: {e}")
                            except Exception as e:
                                logger.error(f"Error processing KPP recording: {e}")
                        else:
                            logger.info(f"Speech too short ({speech_duration:.2f}s), ignoring")

                        # Сброс состояния
                        is_recording = False
                        audio_buffer = []

                        # Небольшая задержка перед следующим обнаружением
                        time.sleep(0.5)
                    else:
                        # Тишина еще недостаточно длинная, продолжаем запись
                        audio_buffer.append(audio_block)

            # Короткая пауза для снижения нагрузки на CPU
            time.sleep(0.01)

        # Остановка и закрытие потока
        stream.stop()
        stream.close()
        logger.info("Audio stream closed")

    except Exception as e:
        logger.error(f"Error in KPP mode worker: {e}")
    finally:
        logger.info("KPP mode worker stopped")