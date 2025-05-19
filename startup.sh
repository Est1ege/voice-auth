#!/bin/bash
# startup.sh - скрипт для запуска системы голосовой аутентификации

set -e

# Цвета для вывода
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}====================================================${NC}"
echo -e "${BLUE}    Система голосовой аутентификации с защитой      ${NC}"
echo -e "${BLUE}              от спуфинг-атак                       ${NC}"
echo -e "${BLUE}====================================================${NC}"

# Проверка наличия Docker и Docker Compose
if ! command -v docker &> /dev/null; then
    echo -e "${RED}Docker не установлен. Установите Docker для продолжения.${NC}"
    exit 1
fi

if ! command -v docker-compose &> /dev/null; then
    echo -e "${RED}Docker Compose не установлен. Установите Docker Compose для продолжения.${NC}"
    exit 1
fi

# Проверка наличия .env файла
if [ ! -f .env ]; then
    echo -e "${YELLOW}Файл .env не найден. Создаем с настройками по умолчанию...${NC}"
    cat > .env << EOL
SECRET_KEY=your_secret_key_here
DB_USER=mongo
DB_PASSWORD=password123
EOL
    echo -e "${GREEN}Файл .env создан. Рекомендуется изменить пароли перед использованием в продакшн.${NC}"
fi

# Создание необходимых директорий
echo -e "${BLUE}Создание необходимых директорий...${NC}"
mkdir -p shared/models shared/embeddings shared/audio nginx/conf nginx/ssl static

# Копирование конфигурации Nginx
echo -e "${BLUE}Настройка Nginx...${NC}"
if [ ! -f nginx/conf/default.conf ]; then
    cat > nginx/conf/default.conf << EOL
server {
    listen 80;
    server_name localhost;

    location / {
        proxy_pass http://web:8000;
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
    }
}
EOL
    echo -e "${GREEN}Конфигурация Nginx создана.${NC}"
fi

# Запуск системы
echo -e "${BLUE}Запуск системы голосовой аутентификации...${NC}"
docker-compose up -d

# Проверка статуса запуска
echo -e "${BLUE}Проверка статуса контейнеров...${NC}"
sleep 5
docker-compose ps

# Инструкции по использованию
echo -e "${GREEN}====================================================${NC}"
echo -e "${GREEN}    Система голосовой аутентификации запущена      ${NC}"
echo -e "${GREEN}====================================================${NC}"
echo -e "${YELLOW}Веб-интерфейс доступен по адресу:${NC} http://localhost"
echo -e "${YELLOW}По умолчанию создан пользователь:${NC} admin/admin_password"
echo -e "\n${BLUE}Для мониторинга логов:${NC} docker-compose logs -f"
echo -e "${BLUE}Для остановки системы:${NC} docker-compose down"
echo -e "${BLUE}Для полной остановки с удалением данных:${NC} docker-compose down -v"
echo -e "${GREEN}====================================================${NC}"