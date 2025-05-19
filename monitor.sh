#!/bin/bash
# monitor.sh - Скрипт для мониторинга системы голосовой аутентификации

set -e

# Цвета для вывода
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}====================================================${NC}"
echo -e "${BLUE}        Мониторинг системы аутентификации           ${NC}"
echo -e "${BLUE}====================================================${NC}"

# Проверка наличия Docker
if ! command -v docker &> /dev/null; then
    echo -e "${RED}Docker не установлен. Установите Docker для продолжения.${NC}"
    exit 1
fi

# Функция для проверки статуса контейнеров
check_containers() {
    echo -e "${BLUE}Проверка статуса контейнеров...${NC}"
    docker-compose ps
    
    # Проверка статуса каждого контейнера
    containers=("nginx" "web" "api" "ml_model" "audio_processor" "db")
    all_running=true
    
    for container in "${containers[@]}"; do
        status=$(docker-compose ps -q $container | xargs docker inspect -f '{{.State.Status}}' 2>/dev/null || echo "not running")
        if [ "$status" != "running" ]; then
            echo -e "${RED}Контейнер $container не запущен!${NC}"
            all_running=false
        fi
    done
    
    if $all_running; then
        echo -e "${GREEN}Все контейнеры запущены и работают.${NC}"
    else
        echo -e "${YELLOW}Некоторые контейнеры не запущены. Проверьте логи для диагностики.${NC}"
    fi
}

# Функция для просмотра логов контейнера
view_logs() {
    container=$1
    lines=$2
    
    echo -e "${BLUE}Последние $lines строк логов контейнера $container:${NC}"
    docker-compose logs --tail=$lines $container
}

# Функция для проверки здоровья API
check_api_health() {
    echo -e "${BLUE}Проверка здоровья API...${NC}"
    health_status=$(curl -s -o /dev/null -w "%{http_code}" http://localhost/api/health || echo "failed")
    
    if [ "$health_status" == "200" ]; then
        echo -e "${GREEN}API работает нормально.${NC}"
    else
        echo -e "${RED}API не отвечает или отвечает с ошибкой ($health_status)!${NC}"
    fi
}

# Функция для проверки использования ресурсов
check_resources() {
    echo -e "${BLUE}Использование ресурсов контейнерами:${NC}"
    docker stats --no-stream $(docker-compose ps -q)
}

# Функция для проверки дискового пространства
check_disk_space() {
    echo -e "${BLUE}Проверка дискового пространства для томов Docker:${NC}"
    docker system df
    
    echo -e "${BLUE}Проверка дискового пространства для системы:${NC}"
    df -h | grep -E "Filesystem|/$"
}

# Меню выбора действия
show_menu() {
    echo -e "\n${YELLOW}Выберите действие:${NC}"
    echo "1) Проверить статус контейнеров"
    echo "2) Просмотреть логи API"
    echo "3) Просмотреть логи ML-сервиса"
    echo "4) Просмотреть логи веб-интерфейса"
    echo "5) Просмотреть логи обработки аудио"
    echo "6) Проверить здоровье API"
    echo "7) Проверить использование ресурсов"
    echo "8) Проверить дисковое пространство"
    echo "9) Перезапустить проблемные контейнеры"
    echo "0) Выход"
    
    read -p "Введите номер действия: " choice
    
    case $choice in
        1) check_containers ;;
        2) view_logs api 50 ;;
        3) view_logs ml_model 50 ;;
        4) view_logs web 50 ;;
        5) view_logs audio_processor 50 ;;
        6) check_api_health ;;
        7) check_resources ;;
        8) check_disk_space ;;
        9) 
            docker-compose restart
            echo -e "${GREEN}Контейнеры перезапущены.${NC}"
            ;;
        0) 
            echo -e "${GREEN}Выход из мониторинга.${NC}"
            exit 0
            ;;
        *)
            echo -e "${RED}Некорректный выбор!${NC}"
            ;;
    esac
    
    echo -e "\nНажмите Enter для продолжения..."
    read
}

# Основной цикл
while true; do
    clear
    echo -e "${BLUE}====================================================${NC}"
    echo -e "${BLUE}        Мониторинг системы аутентификации           ${NC}"
    echo -e "${BLUE}====================================================${NC}"
    show_menu
done