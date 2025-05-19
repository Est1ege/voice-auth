FROM python:3.9-slim

WORKDIR /app

# Установка зависимостей
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    libsndfile1 \
    libsndfile1-dev \
    && rm -rf /var/lib/apt/lists/*

# Создаем необходимые директории
RUN mkdir -p /app/logs /shared/audio

# Копирование и установка зависимостей
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Явно установить совместимую версию pymongo и motor
# Motor 3.1.1 требует pymongo 4.5.0 или более новую версию
RUN pip install --no-cache-dir pymongo==4.5.0 motor==3.1.1

# Копирование кода приложения
COPY . .

# Команда запуска
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "5000"]