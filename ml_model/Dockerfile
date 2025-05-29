# Улучшенный Dockerfile
FROM pytorch/pytorch:2.0.0-cuda11.7-cudnn8-runtime

ENV TZ=Asia/Almaty
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone


WORKDIR /app

# Установка системных зависимостей
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    libsndfile1 \
    libsndfile1-dev \
    python3-dev \
    git \
    ffmpeg \
    sox \
    libsox-fmt-all \
    tzdata \
    locales \
    && rm -rf /var/lib/apt/lists/*

RUN sed -i -e 's/# en_US.UTF-8 UTF-8/en_US.UTF-8 UTF-8/' /etc/locale.gen && \
    dpkg-reconfigure --frontend=noninteractive locales && \
    update-locale LANG=en_US.UTF-8

ENV LANG=en_US.UTF-8
ENV LANGUAGE=en_US:en
ENV LC_ALL=en_US.UTF-8

# Создание директорий для логов и моделей
RUN mkdir -p /app/logs /shared/models /shared/embeddings /shared/audio /shared/temp

# Фиксируем проблемы с зависимостями
RUN pip uninstall -y typing-extensions anyio
RUN pip install --no-cache-dir typing-extensions==4.7.1 anyio==3.7.1

# Установка PyTorch с оптимизированной версией
RUN pip install torch==2.1.0+cu118 torchvision==0.16.0+cu118 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu118

# Установка SpeechBrain с правильной версией
RUN pip install --no-cache-dir speechbrain==0.5.15

# Установка библиотек для работы с аудио
RUN pip install --no-cache-dir \
    librosa==0.10.0 \
    soundfile==0.12.1 \
    audioread \
    resampy \
    pydub \
    ffmpeg-python

# Установка основных зависимостей
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Оптимизация CUDA
ENV CUDA_LAUNCH_BLOCKING=1
ENV TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;8.6"
ENV PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:128"

# Увеличение лимитов памяти для работы с аудио
ENV MALLOC_TRIM_THRESHOLD_=0
ENV MALLOC_MMAP_THRESHOLD_=131072

# Копирование кода приложения
COPY . .

# Запуск сервера
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "5000", "--workers", "1"]