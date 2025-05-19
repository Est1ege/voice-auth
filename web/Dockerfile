# web/Dockerfile 
FROM python:3.9-slim 
 
WORKDIR /app 
 
# Установка необходимых зависимостей 
RUN apt-get update && apt-get install -y --no-install-recommends \ 
    gcc \ 
    g++ \ 
    && rm -rf /var/lib/apt/lists/* 
 
# Установка Python-библиотек 
COPY requirements.txt . 
RUN pip install --no-cache-dir -r requirements.txt 
 
# Создание директорий для шаблонов и статических файлов 
RUN mkdir -p /app/templates /app/static /app/templates/admin /app/templates/user 
 
# Копирование кода приложения и шаблонов 
COPY . . 
 
# Запуск веб-сервера 
CMD ["flask", "run", "--host=0.0.0.0", "--port=8000"] 
