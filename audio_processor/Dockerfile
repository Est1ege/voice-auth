FROM python:3.9-slim

# Исправление флага
RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 \
    ffmpeg \
    gcc \
    libportaudio2 \
    libasound2-dev \
    g++ \
    portaudio19-dev \
    && rm -rf /var/lib/apt/lists/*

# Прочие команды...

# Создание директорий для приложения и логов
WORKDIR /app
RUN mkdir -p /app/logs /shared/audio /shared/audio/temp

# Копирование зависимостей
COPY requirements.txt .
# Установка критических зависимостей перед sounddevice
RUN pip install --no-cache-dir numpy cffi
RUN pip install --no-cache-dir -r requirements.txt
# Установка sounddevice с дополнительными опциями
RUN pip install --no-cache-dir sounddevice --no-binary=sounddevice

# Копирование кода приложения
COPY . .

# Указание порта
EXPOSE 5000

# Запуск сервиса
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "5000"]