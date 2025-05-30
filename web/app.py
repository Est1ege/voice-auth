# web/app.py
from flask import Flask, render_template, request, jsonify, session, redirect, url_for, flash, send_from_directory, Response
import os
import requests
import json
import datetime
import uuid
import tempfile
import zipfile
import shutil
import psutil
import time
from datetime import datetime, timedelta
from functools import wraps
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
import docker

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'dev_key_123')
API_URL = os.environ.get('API_URL', 'http://api:5000')
UPLOAD_FOLDER = os.environ.get('UPLOAD_FOLDER', '/shared/temp')
EXPORT_FOLDER = os.environ.get('EXPORT_FOLDER', '/shared/exports')

# Создаем директории, если они не существуют
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(EXPORT_FOLDER, exist_ok=True)

# Допустимые расширения файлов
ALLOWED_EXTENSIONS = {'wav', 'zip'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

requests.post

# Имитация базы данных пользователей для демонстрации
# В реальной системе должно быть в базе данных
USERS = {
    'admin': {
        'password': generate_password_hash('admin_password'),
        'role': 'admin'
    }
}

# Декоратор для проверки аутентификации
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'username' not in session:
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

# Декоратор для проверки прав администратора
def admin_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'username' not in session or USERS.get(session['username'], {}).get('role') != 'admin':
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

@app.route('/')
def index():
    if 'username' in session:
        if USERS.get(session['username'], {}).get('role') == 'admin':
            return redirect(url_for('admin_dashboard'))
        return redirect(url_for('user_dashboard'))
    return redirect(url_for('login'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    error = None
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        if username in USERS and check_password_hash(USERS[username]['password'], password):
            session['username'] = username
            if USERS[username]['role'] == 'admin':
                return redirect(url_for('admin_dashboard'))
            return redirect(url_for('user_dashboard'))
        else:
            error = 'Incorrect username or password'

    return render_template('login.html', error=error)

@app.route('/logout')
def logout():
    session.pop('username', None)
    return redirect(url_for('login'))


# Замените функцию admin_dashboard в web/app.py

@app.route('/admin')
@admin_required
def admin_dashboard():
    """Главная страница администратора с актуальными данными"""
    dashboard_data = {
        'total_users': 0,
        'active_users': 0,
        'entries_today': 0,
        'spoofing_attempts': 0,
        'recent_events': [],
        'connection_error': False
    }

    try:
        app.logger.info("Loading dashboard data...")

        # 1. Получаем системный статус для данных о пользователях
        try:
            status_response = requests.get(f"{API_URL}/api/system/status", timeout=10)
            if status_response.status_code == 200:
                system_status = status_response.json()
                dashboard_data['total_users'] = system_status.get('users_count', 0)
                dashboard_data['active_users'] = system_status.get('active_users_count', 0)

                # Используем статистику за сегодня, если она есть
                today_stats = system_status.get('today_stats', {})
                dashboard_data['entries_today'] = today_stats.get('successful_auths', 0)
                dashboard_data['spoofing_attempts'] = today_stats.get('spoofing_attempts', 0)

                app.logger.info(
                    f"System status: users={dashboard_data['total_users']}, active={dashboard_data['active_users']}")
            else:
                app.logger.warning(f"System status API returned: {status_response.status_code}")
        except requests.exceptions.RequestException as e:
            app.logger.error(f"Error getting system status: {e}")
            dashboard_data['connection_error'] = True

        # 2. Получаем логи для Recent Events
        try:
            logs_response = requests.get(f"{API_URL}/api/logs", params={"limit": 10}, timeout=10)
            if logs_response.status_code == 200:
                logs_data = logs_response.json()
                logs = logs_data.get('logs', [])

                app.logger.info(f"Retrieved {len(logs)} log entries")

                # Обрабатываем логи для отображения
                recent_events = []
                for log in logs:
                    try:
                        event = {
                            'timestamp': format_timestamp(log.get('timestamp', '')),
                            'type': get_event_display_name(log.get('event_type', '')),
                            'user': log.get('user_name', 'Unknown'),
                            'status': 'Success' if log.get('success', False) else 'Failed'
                        }
                        recent_events.append(event)
                    except Exception as e:
                        app.logger.warning(f"Error processing log entry: {e}")
                        continue

                dashboard_data['recent_events'] = recent_events
                app.logger.info(f"Processed {len(recent_events)} events for display")
            else:
                app.logger.warning(f"Logs API returned: {logs_response.status_code}")
        except requests.exceptions.RequestException as e:
            app.logger.error(f"Error getting logs: {e}")

        # 3. Если данные о событиях за сегодня не получены из системного статуса,
        # подсчитываем их из логов
        if dashboard_data['entries_today'] == 0 and dashboard_data['spoofing_attempts'] == 0:
            try:
                from datetime import date
                today = date.today()

                for log in logs_data.get('logs', []) if 'logs_data' in locals() else []:
                    try:
                        log_date_str = log.get('timestamp', '')
                        if log_date_str:
                            # Извлекаем дату из timestamp
                            log_date = None
                            for fmt in ['%Y-%m-%dT%H:%M:%S.%f', '%Y-%m-%dT%H:%M:%S']:
                                try:
                                    log_date = datetime.strptime(log_date_str.split('Z')[0], fmt).date()
                                    break
                                except:
                                    continue

                            if log_date == today:
                                event_type = log.get('event_type', '')
                                if event_type == 'authorization_successful':
                                    dashboard_data['entries_today'] += 1
                                elif event_type == 'spoofing_attempt':
                                    dashboard_data['spoofing_attempts'] += 1
                    except Exception as e:
                        continue

                app.logger.info(
                    f"Calculated from logs: entries_today={dashboard_data['entries_today']}, spoofing_attempts={dashboard_data['spoofing_attempts']}")
            except Exception as e:
                app.logger.warning(f"Error calculating today's stats from logs: {e}")

        app.logger.info(f"Final dashboard data: {dashboard_data}")

    except Exception as e:
        app.logger.error(f"Unexpected error loading dashboard: {e}")
        dashboard_data['connection_error'] = True

    return render_template('admin/dashboard.html',
                           username=session['username'],
                           **dashboard_data)

def format_timestamp(timestamp_str):
    """Форматирует временную метку для отображения"""
    if not timestamp_str:
        return 'Unknown'

    try:
        # Пробуем разные форматы
        for fmt in ['%Y-%m-%dT%H:%M:%S.%f', '%Y-%m-%dT%H:%M:%S']:
            try:
                dt = datetime.strptime(timestamp_str.split('Z')[0], fmt)
                return dt.strftime('%Y-%m-%d %H:%M:%S')
            except:
                continue
        # Если не получилось распарсить, возвращаем как есть
        return timestamp_str[:19].replace('T', ' ')
    except Exception:
        return timestamp_str

def get_event_display_name(event_type):
    """Преобразует тип события в читаемое название"""
    event_names = {
        'authorization_successful': 'Successful Login',
        'authorization_attempt': 'Failed Login',
        'spoofing_attempt': 'Spoofing Attempt',
        'user_created': 'User Created',
        'user_activated': 'User Activated',
        'user_deleted': 'User Deleted',
        'training_started': 'Training Started',
        'model_deployed': 'Model Deployed',
        'voice_sample_added': 'Voice Sample Added'
    }
    return event_names.get(event_type, event_type.replace('_', ' ').title())

@app.route('/user')
@login_required
def user_dashboard():
    return render_template('user/dashboard.html', username=session['username'])

@app.route('/admin/users')
@admin_required
def manage_users():
    # Получение списка пользователей из API
    try:
        response = requests.get(f"{API_URL}/api/users")
        users = response.json().get('users', [])
    except:
        users = []

    # Параметр show_quick_links=True активирует отображение кнопок для новых функций
    return render_template('admin/users.html', users=users, show_quick_links=True)

# web/app.py - обновите маршрут add_user
@app.route('/admin/users/new', methods=['GET', 'POST'])
@admin_required
def add_user():
    if request.method == 'POST':
        user_data = {
            'name': request.form['name'],
            'email': request.form['email'],
            'role': request.form['role'],
            'department': request.form.get('department', ''),
            'position': request.form.get('position', ''),
            'access_level': request.form.get('access_level', 'standard'),
            'active': False
        }

        # Отправка данных в API
        try:
            response = requests.post(f"{API_URL}/api/users", json=user_data)
            if response.status_code == 201:
                user_id = response.json().get('user_id')
                return redirect(url_for('enroll_user_voice', user_id=user_id))
            else:
                # Improved error handling
                error_response = response.json()
                error_detail = error_response.get('detail', error_response.get('message', 'Unknown error'))
                error = f"Error creating user: {error_detail}"
                return render_template('admin/add_user.html', error=error)
        except Exception as e:
            error = f"Error connecting to API: {str(e)}"
            return render_template('admin/add_user.html', error=error)

    return render_template('admin/add_user.html')


# Замените существующую функцию api_proxy в web/app.py на эту обновленную версию

@app.route('/api-proxy/<path:subpath>', methods=['GET', 'POST', 'PUT', 'DELETE'])
@admin_required
def api_proxy(subpath):
    """
    Прокси для API запросов к серверу API.
    Поддерживает все HTTP методы для полной функциональности.
    """
    try:
        # Создаем полный URL для запроса к API
        api_url = f"{API_URL}/api/{subpath}"

        # Определяем метод запроса
        method = request.method

        # Подготавливаем параметры для запроса
        request_kwargs = {
            'params': request.args,
            'headers': {key: value for key, value in request.headers if key.lower() not in ['host', 'content-length']},
            'stream': True
        }

        # Добавляем данные для POST/PUT запросов
        if method in ['POST', 'PUT'] and request.data:
            request_kwargs['data'] = request.data

        # Добавляем JSON данные если есть
        if method in ['POST', 'PUT'] and request.is_json:
            request_kwargs['json'] = request.get_json()
            request_kwargs.pop('data', None)  # Убираем data если используем json

        # Отправляем запрос к API
        if method == 'GET':
            response = requests.get(api_url, **request_kwargs)
        elif method == 'POST':
            response = requests.post(api_url, **request_kwargs)
        elif method == 'PUT':
            response = requests.put(api_url, **request_kwargs)
        elif method == 'DELETE':
            response = requests.delete(api_url, **request_kwargs)
        else:
            return Response("Method not allowed", status=405)

        # Создаем объект Response с данными из API
        if response.headers.get('content-type', '').startswith('application/json'):
            # Для JSON ответов
            flask_response = Response(
                response=response.content,
                status=response.status_code,
                content_type='application/json'
            )
        else:
            # Для файлов и других типов контента
            flask_response = Response(
                response=response.content,
                status=response.status_code
            )

        # Копируем заголовки из ответа API
        for key, value in response.headers.items():
            if key.lower() not in ('content-length', 'connection', 'content-encoding', 'transfer-encoding'):
                flask_response.headers[key] = value

        return flask_response

    except requests.exceptions.ConnectionError:
        app.logger.error(f"Connection error to API server for {method} {subpath}")
        return Response("API server unavailable", status=503)
    except requests.exceptions.Timeout:
        app.logger.error(f"Timeout error to API server for {method} {subpath}")
        return Response("API server timeout", status=504)
    except Exception as e:
        app.logger.error(f"Error in API proxy for {method} {subpath}: {str(e)}")
        return Response(f"Proxy error: {str(e)}", status=500)

@app.route('/admin/users/<user_id>/upload_photo', methods=['POST'])
@admin_required
def upload_user_photo(user_id):
    """Загрузка фотографии пользователя"""
    try:
        if 'photo' not in request.files:
            flash('No photo selected', 'danger')
            return redirect(url_for('enroll_user_voice', user_id=user_id))

        photo = request.files['photo']

        if photo.filename == '':
            flash('No photo selected', 'danger')
            return redirect(url_for('enroll_user_voice', user_id=user_id))

        # Проверка типа файла
        allowed_extensions = {'jpg', 'jpeg', 'png'}
        if not photo.filename.lower().split('.')[-1] in allowed_extensions:
            flash('Invalid file format. Only JPG and PNG are allowed.', 'danger')
            return redirect(url_for('enroll_user_voice', user_id=user_id))

        # Получение информации о пользователе
        response = requests.get(f"{API_URL}/api/users/{user_id}")

        if response.status_code != 200:
            flash('User not found', 'danger')
            return redirect(url_for('manage_users'))

        # Сохранение фото
        photo_path = os.path.join(UPLOAD_FOLDER, f"photo_{user_id}.jpg")
        photo.save(photo_path)

        # Отправка фото в API
        with open(photo_path, 'rb') as f:
            upload_response = requests.post(
                f"{API_URL}/api/users/{user_id}/photo",
                files={'photo': f}
            )

        if upload_response.status_code == 200:
            flash('Photo uploaded successfully', 'success')
        else:
            flash(f'Error uploading photo: {upload_response.json().get("detail", "Unknown error")}', 'danger')

        # Удаление временного файла
        if os.path.exists(photo_path):
            os.remove(photo_path)

        return redirect(url_for('enroll_user_voice', user_id=user_id))
    except Exception as e:
        flash(f'Error: {str(e)}', 'danger')
        return redirect(url_for('enroll_user_voice', user_id=user_id))


@app.route('/admin/users/<user_id>/enroll', methods=['GET', 'POST'])
@admin_required
def enroll_user_voice(user_id):
    # Получение информации о пользователе
    try:
        response = requests.get(f"{API_URL}/api/users/{user_id}")
        user = response.json().get('user', {})
    except:
        user = {'id': user_id, 'name': 'Unknown user'}

    # Получение прогресса записи голоса
    try:
        response = requests.get(f"{API_URL}/api/users/{user_id}/voice_samples")
        voice_samples = response.json().get('samples', [])
        progress = len(voice_samples)
        total_required = 5  # Требуемое количество образцов
    except:
        voice_samples = []
        progress = 0
        total_required = 5

    if request.method == 'POST':
        if 'audio_data' not in request.files:
            return jsonify({'error': 'No audio file uploaded'}), 400

        audio_file = request.files['audio_data']

        # Отправка аудиофайла в API
        try:
            files = {'audio': (f"{uuid.uuid4()}.wav", audio_file)}
            response = requests.post(
                f"{API_URL}/api/users/{user_id}/voice_samples",
                files=files
            )

            return jsonify(response.json())
        except Exception as e:
            return jsonify({'error': str(e)}), 500

    return render_template(
        'admin/enroll_voice.html',
        user=user,
        progress=progress,
        total_required=total_required
    )

@app.route('/admin/users/<user_id>/activate', methods=['POST'])
@admin_required
def activate_user(user_id):
    try:
        response = requests.post(f"{API_URL}/api/users/{user_id}/activate")
        return jsonify(response.json())
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/admin/logs')
@admin_required
def access_logs():
    # Получение журналов доступа из API
    try:
        response = requests.get(f"{API_URL}/api/logs")
        logs = response.json().get('logs', [])
    except:
        logs = []

    return render_template('admin/logs.html', logs=logs)

# Изменения в web/app.py, в маршруте system_status
# Замените существующий маршрут system_status в web/app.py

@app.route('/admin/system')
@admin_required
def system_status():
    """Страница статуса системы с реальными данными"""
    try:
        # Получаем базовый статус системы
        response = requests.get(f"{API_URL}/api/system/status", timeout=10)

        if response.status_code == 200:
            status = response.json()

            # Добавляем дополнительную информацию
            status = enhance_system_status(status)
        else:
            app.logger.error(f"API returned status code: {response.status_code}")
            status = get_default_status()

    except requests.exceptions.ConnectionError:
        app.logger.error("Cannot connect to API service")
        status = get_default_status()
        status['api_status'] = 'error'
        status['connection_error'] = True

    except requests.exceptions.Timeout:
        app.logger.error("API request timeout")
        status = get_default_status()
        status['api_status'] = 'timeout'

    except Exception as e:
        app.logger.error(f"Unexpected error getting system status: {e}")
        status = get_default_status()
        status['api_status'] = 'error'
        status['error_message'] = str(e)

    return render_template('admin/system.html', status=status, show_transfer_link=True)

def enhance_system_status(base_status):
    """Дополняет базовый статус системы дополнительной информацией"""
    try:
        # Получаем реальную информацию о хранилище
        storage_info = get_storage_usage()
        base_status['storage'] = storage_info

        # Получаем информацию о контейнерах Docker (если доступно)
        containers_info = get_docker_containers_info()
        base_status['containers'] = containers_info

        # Получаем информацию о бэкапах (заглушка, так как система бэкапов не реализована)
        base_status['backups'] = []
        base_status['backup_schedule'] = 'Daily at 02:00'

        # Добавляем версию API
        base_status['api_version'] = '1.0.0'

    except Exception as e:
        app.logger.warning(f"Error enhancing system status: {e}")

    return base_status

def get_storage_usage():
    """Получает реальную информацию об использовании хранилища"""
    import shutil

    storage = {
        'audio_used': '0 MB',
        'audio_total': '1 GB',
        'audio_percent': 0,
        'db_used': '0 MB',
        'db_total': '1 GB',
        'db_percent': 0,
        'ml_used': '0 MB',
        'ml_total': '1 GB',
        'ml_percent': 0
    }

    try:
        # Проверяем использование диска для аудио файлов
        audio_path = '/shared/audio'
        if os.path.exists(audio_path):
            audio_size = get_directory_size(audio_path)
            storage['audio_used'] = format_bytes(audio_size)

            # Получаем общий размер диска
            total, used, free = shutil.disk_usage('/shared')
            storage['audio_total'] = format_bytes(total)
            storage['audio_percent'] = min(100, int((audio_size / total) * 100))

        # Проверяем использование для моделей ML
        models_path = '/shared/models'
        if os.path.exists(models_path):
            models_size = get_directory_size(models_path)
            storage['ml_used'] = format_bytes(models_size)
            storage['ml_total'] = format_bytes(total)
            storage['ml_percent'] = min(100, int((models_size / total) * 100))

        # Для БД используем общее использование диска
        storage['db_used'] = format_bytes(used)
        storage['db_total'] = format_bytes(total)
        storage['db_percent'] = min(100, int((used / total) * 100))

    except Exception as e:
        app.logger.warning(f"Error getting storage usage: {e}")

    return storage

def get_directory_size(path):
    """Вычисляет размер директории в байтах"""
    total_size = 0
    try:
        for dirpath, dirnames, filenames in os.walk(path):
            for filename in filenames:
                filepath = os.path.join(dirpath, filename)
                try:
                    total_size += os.path.getsize(filepath)
                except OSError:
                    pass
    except Exception:
        pass
    return total_size

def format_bytes(bytes_value):
    """Форматирует размер в байтах в читаемый формат"""
    if bytes_value == 0:
        return "0 B"

    sizes = ['B', 'KB', 'MB', 'GB', 'TB']
    i = 0
    while bytes_value >= 1024 and i < len(sizes) - 1:
        bytes_value /= 1024.0
        i += 1

    return f"{bytes_value:.1f} {sizes[i]}"

def get_docker_containers_info():
    """Получает реальную информацию о Docker контейнерах"""
    containers = []

    try:
        # Подключение к Docker API
        client = docker.from_env()

        # Получаем все контейнеры проекта (по префиксу имени или label)
        project_containers = client.containers.list(all=True)

        # Фильтруем контейнеры нашего проекта
        voice_auth_containers = [
            container for container in project_containers
            if
            any(keyword in container.name.lower() for keyword in ['voice', 'auth', 'api', 'ml', 'web', 'db', 'mongo'])
        ]

        for container in voice_auth_containers:
            try:
                # Получаем базовую информацию
                container_info = {
                    'name': container.name,
                    'status': container.status.title(),
                    'cpu': '0%',
                    'memory': '0 MB',
                    'network': '0 KB/s',
                    'uptime': 'Unknown'
                }

                # Если контейнер запущен, получаем детальную статистику
                if container.status == 'running':
                    # Получаем статистику использования ресурсов
                    stats = container.stats(stream=False)

                    # CPU использование
                    cpu_percent = calculate_cpu_percent(stats)
                    container_info['cpu'] = f"{cpu_percent:.1f}%"

                    # Память
                    memory_usage = stats['memory_stats'].get('usage', 0)
                    memory_limit = stats['memory_stats'].get('limit', 0)

                    if memory_usage > 0:
                        memory_mb = memory_usage / (1024 * 1024)
                        container_info['memory'] = f"{memory_mb:.0f} MB"

                    # Сетевой трафик
                    network_io = calculate_network_io(stats)
                    container_info['network'] = network_io

                    # Время работы
                    uptime = calculate_uptime(container)
                    container_info['uptime'] = uptime

                containers.append(container_info)

            except Exception as e:
                # Если не удалось получить статистику для конкретного контейнера
                containers.append({
                    'name': container.name,
                    'status': container.status.title(),
                    'cpu': 'N/A',
                    'memory': 'N/A',
                    'network': 'N/A',
                    'uptime': 'N/A'
                })
                print(f"Warning: Could not get stats for container {container.name}: {e}")

    except docker.errors.DockerException as e:
        print(f"Docker API error: {e}")
        # Возвращаем пустой список, если Docker недоступен
        return []
    except Exception as e:
        print(f"Error getting Docker containers info: {e}")
        return []

    return containers

def calculate_cpu_percent(stats):
    """Вычисляет процент использования CPU контейнером"""
    try:
        # Docker CPU статистика
        cpu_stats = stats.get('cpu_stats', {})
        precpu_stats = stats.get('precpu_stats', {})

        cpu_usage = cpu_stats.get('cpu_usage', {})
        precpu_usage = precpu_stats.get('cpu_usage', {})

        cpu_total = cpu_usage.get('total_usage', 0)
        precpu_total = precpu_usage.get('total_usage', 0)

        cpu_system = cpu_stats.get('system_cpu_usage', 0)
        precpu_system = precpu_stats.get('system_cpu_usage', 0)

        cpu_num = len(cpu_usage.get('percpu_usage', []))
        if cpu_num == 0:
            cpu_num = psutil.cpu_count()

        cpu_delta = cpu_total - precpu_total
        system_delta = cpu_system - precpu_system

        if system_delta > 0 and cpu_delta > 0:
            cpu_percent = (cpu_delta / system_delta) * cpu_num * 100.0
            return cpu_percent

    except Exception as e:
        print(f"Error calculating CPU percent: {e}")

    return 0.0

def calculate_network_io(stats):
    """Вычисляет сетевой I/O"""
    try:
        networks = stats.get('networks', {})
        if not networks:
            return '0 KB/s'

        total_rx = 0
        total_tx = 0

        for interface, data in networks.items():
            total_rx += data.get('rx_bytes', 0)
            total_tx += data.get('tx_bytes', 0)

        # Возвращаем суммарный трафик в KB/s (упрощенно)
        total_bytes = total_rx + total_tx
        if total_bytes < 1024:
            return f"{total_bytes} B/s"
        elif total_bytes < 1024 * 1024:
            return f"{total_bytes / 1024:.1f} KB/s"
        else:
            return f"{total_bytes / (1024 * 1024):.1f} MB/s"

    except Exception as e:
        print(f"Error calculating network I/O: {e}")
        return '0 KB/s'

def calculate_uptime(container):
    """Вычисляет время работы контейнера"""
    try:
        # Получаем время запуска из атрибутов контейнера
        container.reload()  # Обновляем информацию о контейнере

        started_at = container.attrs['State']['StartedAt']

        # Парсим время запуска
        # Формат: "2024-01-15T10:30:45.123456789Z"
        started_time = datetime.fromisoformat(started_at.replace('Z', '+00:00'))

        # Убираем timezone info для упрощения вычислений
        started_time = started_time.replace(tzinfo=None)
        now = datetime.utcnow()

        uptime_delta = now - started_time

        # Форматируем время работы
        days = uptime_delta.days
        hours, remainder = divmod(uptime_delta.seconds, 3600)
        minutes, _ = divmod(remainder, 60)

        if days > 0:
            return f"{days}d {hours}h"
        elif hours > 0:
            return f"{hours}h {minutes}m"
        else:
            return f"{minutes}m"

    except Exception as e:
        print(f"Error calculating uptime: {e}")
        return 'Unknown'

def get_real_system_status():
    """Получает реальный системный статус без заглушек"""
    try:
        # Базовая информация о системе
        status = {
            'api_status': 'unknown',
            'ml_status': 'unknown',
            'db_status': 'unknown',
            'users_count': 0,
            'active_users_count': 0,
            'device': get_device_info(),
            'api_version': '1.0.0',
            'storage': get_real_storage_usage(),
            'containers': get_docker_containers_info(),
            'backups': get_backup_status(),
            'backup_schedule': get_backup_schedule(),
            'system_resources': get_detailed_system_resources(),
            'timestamp': datetime.now().isoformat()
        }

        return status

    except Exception as e:
        print(f"Error getting real system status: {e}")
        return get_fallback_status()

def get_device_info():
    """Получает информацию об устройстве"""
    try:
        # Проверяем наличие GPU
        try:
            import torch
            if torch.cuda.is_available():
                gpu_name = torch.cuda.get_device_name(0)
                gpu_count = torch.cuda.device_count()
                return f"GPU: {gpu_name} ({gpu_count} device{'s' if gpu_count > 1 else ''})"
        except ImportError:
            pass

        # Если GPU недоступно, возвращаем информацию о CPU
        cpu_info = f"CPU: {psutil.cpu_count()} cores"
        return cpu_info

    except Exception as e:
        print(f"Error getting device info: {e}")
        return "Unknown"

def get_real_storage_usage():
    """Получает реальное использование хранилища"""
    try:
        storage = {}

        # Общее использование диска
        total, used, free = psutil.disk_usage('/')

        # Использование директории аудио файлов
        audio_path = '/shared/audio'
        if os.path.exists(audio_path):
            audio_size = get_directory_size(audio_path)
            storage['audio_used'] = format_bytes(audio_size)
            storage['audio_percent'] = min(100, int((audio_size / total) * 100))
        else:
            storage['audio_used'] = '0 B'
            storage['audio_percent'] = 0

        # Использование моделей ML
        models_path = '/shared/models'
        if os.path.exists(models_path):
            models_size = get_directory_size(models_path)
            storage['ml_used'] = format_bytes(models_size)
            storage['ml_percent'] = min(100, int((models_size / total) * 100))
        else:
            storage['ml_used'] = '0 B'
            storage['ml_percent'] = 0

        # Общее использование диска
        storage['total_used'] = format_bytes(used)
        storage['total_free'] = format_bytes(free)
        storage['total_size'] = format_bytes(total)
        storage['total_percent'] = int((used / total) * 100)

        return storage

    except Exception as e:
        print(f"Error getting storage usage: {e}")
        return {
            'audio_used': '0 B',
            'audio_percent': 0,
            'ml_used': '0 B',
            'ml_percent': 0,
            'total_used': '0 B',
            'total_free': '0 B',
            'total_size': '0 B',
            'total_percent': 0
        }

def get_backup_status():
    """Получает реальный статус бэкапов"""
    backups = []
    backup_dir = '/shared/backups'

    try:
        if os.path.exists(backup_dir):
            backup_files = [f for f in os.listdir(backup_dir) if f.endswith('.zip')]

            for backup_file in sorted(backup_files, reverse=True)[:10]:  # Последние 10
                file_path = os.path.join(backup_dir, backup_file)
                stat = os.stat(file_path)

                backup_info = {
                    'id': backup_file,
                    'date': datetime.fromtimestamp(stat.st_mtime).strftime('%Y-%m-%d %H:%M'),
                    'type': 'Full' if 'full' in backup_file.lower() else 'Incremental',
                    'size': format_bytes(stat.st_size),
                    'status': 'Success'
                }
                backups.append(backup_info)

    except Exception as e:
        print(f"Error getting backup status: {e}")

    return backups

def get_backup_schedule():
    """Получает расписание бэкапов"""
    try:
        # Проверяем наличие конфигурационного файла
        config_file = '/shared/config/backup_schedule.json'
        if os.path.exists(config_file):
            with open(config_file, 'r') as f:
                config = json.load(f)
                return config.get('schedule', 'Not configured')
        else:
            return 'Daily at 02:00 (default)'
    except Exception as e:
        print(f"Error getting backup schedule: {e}")
        return 'Not configured'

def get_detailed_system_resources():
    """Получает детальную информацию о системных ресурсах"""
    try:
        # CPU информация
        cpu_percent = psutil.cpu_percent(interval=1)
        cpu_count = psutil.cpu_count()
        cpu_freq = psutil.cpu_freq()

        # Память
        memory = psutil.virtual_memory()
        swap = psutil.swap_memory()

        # Диск I/O
        disk_io = psutil.disk_io_counters()

        # Сеть I/O
        net_io = psutil.net_io_counters()

        # Процессы
        process_count = len(psutil.pids())

        return {
            'cpu': {
                'percent': cpu_percent,
                'count': cpu_count,
                'frequency': f"{cpu_freq.current:.0f} MHz" if cpu_freq else "Unknown"
            },
            'memory': {
                'total': format_bytes(memory.total),
                'used': format_bytes(memory.used),
                'free': format_bytes(memory.available),
                'percent': memory.percent
            },
            'swap': {
                'total': format_bytes(swap.total),
                'used': format_bytes(swap.used),
                'free': format_bytes(swap.free),
                'percent': swap.percent
            },
            'disk_io': {
                'read_bytes': format_bytes(disk_io.read_bytes) if disk_io else "0 B",
                'write_bytes': format_bytes(disk_io.write_bytes) if disk_io else "0 B"
            },
            'network_io': {
                'bytes_sent': format_bytes(net_io.bytes_sent),
                'bytes_recv': format_bytes(net_io.bytes_recv)
            },
            'processes': process_count
        }

    except Exception as e:
        print(f"Error getting detailed system resources: {e}")
        return {}

def get_fallback_status():
    """Возвращает базовый статус при ошибках"""
    return {
        'api_status': 'error',
        'ml_status': 'unknown',
        'db_status': 'unknown',
        'users_count': 0,
        'active_users_count': 0,
        'device': 'Unknown',
        'api_version': '1.0.0',
        'storage': {
            'audio_used': '0 B',
            'audio_percent': 0,
            'ml_used': '0 B',
            'ml_percent': 0,
            'total_used': '0 B',
            'total_free': '0 B',
            'total_size': '0 B',
            'total_percent': 0
        },
        'containers': [],
        'backups': [],
        'backup_schedule': 'Not configured',
        'error': True,
        'timestamp': datetime.now().isoformat()
    }

@app.route('/admin/training/cleanup', methods=['POST'])
@admin_required
def cleanup_trainings():
    """Очистка старых задач тренировки"""
    try:
        max_age_days = int(request.form.get('max_age_days', 7))

        response = requests.delete(f"{API_URL}/api/training/cleanup?max_age_days={max_age_days}")

        if response.status_code == 200:
            result = response.json()
            if result.get('success'):
                removed_count = result.get('removed_count', 0)
                flash(f'Удалено {removed_count} старых задач тренировки', 'success')
            else:
                flash(f'Ошибка при очистке: {result.get("message")}', 'danger')
        else:
            flash('Ошибка при очистке задач тренировки', 'danger')

    except ValueError:
        flash('Неверное значение возраста задач', 'danger')
    except Exception as e:
        flash(f'Ошибка при очистке: {str(e)}', 'danger')

    return redirect(url_for('training_list'))

@app.route('/admin/training/dashboard')
@admin_required
def training_dashboard():
    """Главная панель управления тренировками"""
    try:
        # Получаем статистику
        stats_response = requests.get(f"{API_URL}/api/training/stats")
        stats = stats_response.json() if stats_response.status_code == 200 else {}

        # Получаем последние задачи тренировки
        trainings_response = requests.get(f"{API_URL}/api/training/list")
        recent_trainings = []

        if trainings_response.status_code == 200:
            result = trainings_response.json()
            all_trainings = result.get("trainings", [])
            # Берем последние 5 задач
            recent_trainings = all_trainings[:5]

        return render_template(
            'admin/training.html',
            stats=stats,
            recent_trainings=recent_trainings
        )

    except Exception as e:
        error = f"Ошибка при загрузке панели тренировок: {str(e)}"
        return render_template('admin/training.html', error=error)

@app.route('/admin/training/voice-model', methods=['GET', 'POST'])
@admin_required
def train_voice_model():
    if request.method == 'POST':
        # Получение параметров из формы
        batch_size = int(request.form.get('batch_size', 32))
        learning_rate = float(request.form.get('learning_rate', 0.001))
        num_epochs = int(request.form.get('num_epochs', 50))

        # Отправка запроса на тренировку в API
        try:
            response = requests.post(
                f"{API_URL}/api/training/start",
                json={
                    "model_type": "voice_model",
                    "batch_size": batch_size,
                    "learning_rate": learning_rate,
                    "num_epochs": num_epochs
                }
            )

            if response.status_code == 200:
                result = response.json()
                task_id = result.get('task_id')
                return redirect(url_for('training_status', task_id=task_id))
            else:
                error = f"Ошибка при запуске тренировки: {response.json().get('detail')}"
                return render_template('admin/train_voice_model.html', error=error)
        except Exception as e:
            error = f"Ошибка соединения с API: {str(e)}"
            return render_template('admin/train_voice_model.html', error=error)

    return render_template('admin/train_voice_model.html')

@app.route('/admin/training/anti-spoof', methods=['GET', 'POST'])
@admin_required
def train_anti_spoof():
    if request.method == 'POST':
        # Получение параметров из формы
        batch_size = int(request.form.get('batch_size', 32))
        learning_rate = float(request.form.get('learning_rate', 0.001))
        num_epochs = int(request.form.get('num_epochs', 50))

        # Отправка запроса на тренировку в API
        try:
            response = requests.post(
                f"{API_URL}/api/training/start",
                json={
                    "model_type": "anti_spoof",
                    "batch_size": batch_size,
                    "learning_rate": learning_rate,
                    "num_epochs": num_epochs
                }
            )

            if response.status_code == 200:
                result = response.json()
                task_id = result.get('task_id')
                return redirect(url_for('training_status', task_id=task_id))
            else:
                error = f"Ошибка при запуске тренировки: {response.json().get('detail')}"
                return render_template('admin/train_anti_spoof.html', error=error)
        except Exception as e:
            error = f"Ошибка соединения с API: {str(e)}"
            return render_template('admin/train_anti_spoof.html', error=error)

    return render_template('admin/train_anti_spoof.html')

@app.route('/admin/training/<task_id>/status')
@admin_required
def training_status(task_id):
    try:
        response = requests.get(f"{API_URL}/api/training/{task_id}/status")

        if response.status_code == 200:
            status = response.json()
            if status is None:
                return render_template('admin/training_status.html',
                                       error="Задача не найдена в системе",
                                       task_id=task_id)
            if 'progress' not in status or status['progress'] is None:
                status['progress'] = 0
            return render_template('admin/training_status.html', status=status, task_id=task_id)
        elif response.status_code == 404:
            # Добавим редирект на список тренировок, если тренировка не найдена
            flash('Задача тренировки не найдена или завершена. Вот список всех тренировок.', 'warning')
            return redirect(url_for('training_list'))
        else:
            error = f"Ошибка при получении статуса тренировки: {response.json().get('detail', 'Неизвестная ошибка')}"
            return render_template('admin/training_status.html', error=error, task_id=task_id)
    except Exception as e:
        error = f"Ошибка соединения с API: {str(e)}"
        return render_template('admin/training_status.html', error=error, task_id=task_id)

@app.route('/api/voice/system_status')
def voice_system_status():
    """Получение статуса системы голосовой аутентификации"""
    try:
        # Запрос к API сервису для получения статуса
        response = requests.get(f"{API_URL}/api/system/status", timeout=5)

        if response.status_code == 200:
            status_data = response.json()

            # Проверяем основные компоненты системы
            api_working = status_data.get('api_status') == 'ok'
            ml_working = status_data.get('ml_status') == 'ok'

            # Определяем общий статус системы
            if api_working and ml_working:
                system_status = 'ok'
                message = 'System fully operational'
            elif api_working:
                system_status = 'partial'
                message = 'Core system working, some features may be limited'
            else:
                system_status = 'degraded'
                message = 'System experiencing issues'

            return jsonify({
                'api_status': system_status,
                'message': message,
                'anti_spoofing_active': True,  # Предполагаем, что всегда активно
                'core_functional': api_working,
                'details': status_data,
                'timestamp': datetime.now().isoformat()
            })
        else:
            # API недоступен, но система может частично работать
            return jsonify({
                'api_status': 'partial',
                'message': 'Status API unavailable, but core system may be functional',
                'anti_spoofing_active': True,
                'core_functional': True,  # Предполагаем, что базовые функции работают
                'error': f'API returned status {response.status_code}',
                'timestamp': datetime.now().isoformat()
            })

    except requests.exceptions.ConnectionError:
        # Соединение с API недоступно
        return jsonify({
            'api_status': 'partial',
            'message': 'Cannot connect to backend API, but frontend systems operational',
            'anti_spoofing_active': True,  # Интерфейс все еще может работать
            'core_functional': False,
            'error': 'Connection error to backend',
            'timestamp': datetime.now().isoformat()
        })

    except requests.exceptions.Timeout:
        # Тайм-аут запроса
        return jsonify({
            'api_status': 'slow',
            'message': 'System responding slowly',
            'anti_spoofing_active': True,
            'core_functional': True,
            'error': 'Request timeout',
            'timestamp': datetime.now().isoformat()
        })

    except Exception as e:
        # Другие ошибки
        app.logger.error(f"Error getting system status: {e}")
        return jsonify({
            'api_status': 'error',
            'message': 'System status check failed',
            'anti_spoofing_active': False,
            'core_functional': False,
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500

@app.route('/admin/training/list')
@admin_required
def training_list():
    """Отображение списка всех задач тренировки"""
    try:
        # Получаем список всех задач тренировки
        response = requests.get(f"{API_URL}/api/training/list")

        if response.status_code == 200:
            result = response.json()
            all_trainings = result.get("trainings", [])

            # Разделяем задачи по статусам
            active_trainings = [
                task for task in all_trainings
                if task.get('status') in ['starting', 'preparing_data', 'training']
            ]

            completed_trainings = [
                task for task in all_trainings
                if task.get('status') in ['completed', 'error', 'cancelled']
            ]

            # Получаем статистику
            stats_response = requests.get(f"{API_URL}/api/training/stats")
            stats = stats_response.json() if stats_response.status_code == 200 else {}

            return render_template(
                'admin/training_list.html',
                active_trainings=active_trainings,
                completed_trainings=completed_trainings,
                stats=stats
            )
        else:
            error_detail = response.json().get('detail', 'Unknown error') if response.content else 'No response'
            error = f"Ошибка при получении списка задач тренировки: {error_detail}"
            return render_template('admin/training_list.html', error=error)

    except requests.exceptions.ConnectionError:
        error = "Ошибка соединения с API-сервисом"
        return render_template('admin/training_list.html', error=error)
    except Exception as e:
        error = f"Неожиданная ошибка: {str(e)}"
        return render_template('admin/training_list.html', error=error)

@app.route('/admin/training/start', methods=['POST'])
@admin_required
def start_training():
    """Запуск новой задачи тренировки"""
    try:
        model_type = request.form.get('model_type')
        batch_size = int(request.form.get('batch_size', 32))
        learning_rate = float(request.form.get('learning_rate', 0.001))
        num_epochs = int(request.form.get('num_epochs', 50))

        # Валидация параметров
        if model_type not in ['voice_model', 'anti_spoof']:
            flash('Неверный тип модели', 'danger')
            return redirect(url_for('training_dashboard'))

        # Подготовка данных для запроса
        training_data = {
            "model_type": model_type,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "num_epochs": num_epochs
        }

        # Отправка запроса на запуск тренировки
        response = requests.post(
            f"{API_URL}/api/training/start",
            json=training_data
        )

        if response.status_code == 200:
            result = response.json()
            task_id = result.get('task_id')
            flash(f'Тренировка запущена успешно. ID задачи: {task_id}', 'success')
            return redirect(url_for('training_status', task_id=task_id))
        else:
            error_detail = response.json().get('detail', 'Unknown error') if response.content else 'No response'
            flash(f'Ошибка при запуске тренировки: {error_detail}', 'danger')
            return redirect(url_for('training_dashboard'))

    except ValueError as e:
        flash(f'Ошибка в параметрах тренировки: {str(e)}', 'danger')
        return redirect(url_for('training_dashboard'))
    except Exception as e:
        flash(f'Ошибка при запуске тренировки: {str(e)}', 'danger')
        return redirect(url_for('training_dashboard'))

@app.route('/admin/training/<task_id>/stop', methods=['POST'])
@admin_required
def stop_training(task_id):
    """Остановка/отмена задачи тренировки"""
    try:
        response = requests.post(f"{API_URL}/api/training/{task_id}/cancel", timeout=10)

        if response.status_code == 200:
            result = response.json()
            if result.get('success'):
                flash('Тренировка остановлена успешно', 'success')
            else:
                flash(f'Не удалось остановить тренировку: {result.get("message")}', 'warning')
        else:
            flash('Ошибка при остановке тренировки', 'danger')

    except requests.exceptions.ConnectionError:
        flash('Ошибка соединения с API-сервисом', 'danger')
    except requests.exceptions.Timeout:
        flash('Время ожидания ответа истекло', 'danger')
    except Exception as e:
        flash(f'Ошибка при остановке тренировки: {str(e)}', 'danger')

    # Перенаправляем обратно на страницу статуса или список
    return redirect(url_for('training_status', task_id=task_id))

@app.route('/admin/training/<task_id>/deploy', methods=['POST'])
@admin_required
def deploy_model(task_id):
    """Развертывание обученной модели"""
    try:
        # Проверяем статус задачи
        status_response = requests.get(f"{API_URL}/api/training/{task_id}/status", timeout=10)

        if status_response.status_code != 200:
            flash('Не удалось получить статус задачи', 'danger')
            return redirect(url_for('training_status', task_id=task_id))

        status_data = status_response.json()

        if status_data.get('status') != 'completed':
            flash('Модель можно развернуть только после успешного завершения тренировки', 'warning')
            return redirect(url_for('training_status', task_id=task_id))

        # Отправляем запрос на развертывание модели
        deploy_response = requests.post(f"{API_URL}/api/training/{task_id}/deploy", timeout=30)

        if deploy_response.status_code == 200:
            result = deploy_response.json()
            if result.get('success'):
                flash('Модель успешно развернута и готова к использованию', 'success')
            else:
                flash(f'Ошибка при развертывании: {result.get("message")}', 'danger')
        else:
            flash('Ошибка при развертывании модели', 'danger')

    except requests.exceptions.ConnectionError:
        flash('Ошибка соединения с API-сервисом', 'danger')
    except requests.exceptions.Timeout:
        flash('Время ожидания развертывания истекло', 'danger')
    except Exception as e:
        flash(f'Ошибка при развертывании модели: {str(e)}', 'danger')

    return redirect(url_for('training_status', task_id=task_id))

@app.route('/admin/users/simple_add', methods=['GET', 'POST'])
@admin_required
def simple_add_user():
    """Упрощенное добавление пользователя (только ФИО и аудиофайлы)"""
    if request.method == 'POST':
        # Получение данных из формы
        name = request.form.get('name')
        if not name:
            flash('Имя пользователя обязательно', 'danger')
            return render_template('admin/simple_add_user.html', error='Имя пользователя обязательно')

        # Проверка наличия файлов
        if 'audio_files' not in request.files:
            flash('Нет аудиофайлов', 'danger')
            return render_template('admin/simple_add_user.html', error='Нет аудиофайлов')

        audio_files = request.files.getlist('audio_files')
        valid_files = [f for f in audio_files if f and f.filename and allowed_file(f.filename)]

        if len(valid_files) < 5:
            flash(f'Недостаточно аудиофайлов (загружено {len(valid_files)}, требуется минимум 5)', 'danger')
            return render_template('admin/simple_add_user.html', error=f'Недостаточно аудиофайлов (загружено {len(valid_files)}, требуется минимум 5)')

        try:
            # Шаг 1: Создание пользователя через API
            response = requests.post(f"{API_URL}/api/simple/users", data={"name": name})
            response.raise_for_status()

            result = response.json()
            if not result.get("success", False):
                flash('Ошибка при создании пользователя', 'danger')
                return render_template('admin/simple_add_user.html', error='Ошибка при создании пользователя')

            user_id = result.get("user_id")

            # Шаг 2: Загрузка аудиофайлов
            files = []
            for audio_file in valid_files:
                temp_path = os.path.join(UPLOAD_FOLDER, secure_filename(audio_file.filename))
                audio_file.save(temp_path)
                files.append(('audio_files', (os.path.basename(temp_path), open(temp_path, 'rb'), 'audio/wav')))

            # Отправляем запрос на загрузку аудиофайлов
            try:
                upload_response = requests.post(
                    f"{API_URL}/api/simple/upload_voice_batch",
                    data={"user_id": user_id},
                    files=files
                )
                upload_response.raise_for_status()

                # Закрываем все открытые файлы
                for file_item in files:
                    _, (_, file_obj, _) = file_item
                    file_obj.close()

                # Удаляем временные файлы
                for audio_file in valid_files:
                    temp_path = os.path.join(UPLOAD_FOLDER, secure_filename(audio_file.filename))
                    if os.path.exists(temp_path):
                        os.remove(temp_path)

                upload_result = upload_response.json()

                if upload_result.get("success", False):
                    status_message = f"User {name} successfully created with {upload_result.get('processed_files', 0)} audio files"

                    if upload_result.get("activation_status", False):
                        status_message += " and activated"
                    else:
                        status_message += " (not activated)"

                    flash(status_message, 'success')
                    return render_template('admin/simple_add_user.html', success=status_message)
                else:
                    error_message = upload_result.get("message", "Unknown error while loading audio files")
                    flash(error_message, 'danger')
                    return render_template('admin/simple_add_user.html', error=error_message)

            except Exception as e:
                flash(f'Error loading audio files: {str(e)}', 'danger')
                return render_template('admin/simple_add_user.html', error=f'Error loading audio files: {str(e)}')

            finally:
                # Закрываем все открытые файлы при любом исходе
                for file_item in files:
                    try:
                        _, (_, file_obj, _) = file_item
                        file_obj.close()
                    except:
                        pass

                for audio_file in valid_files:
                    temp_path = os.path.join(UPLOAD_FOLDER, secure_filename(audio_file.filename))
                    if os.path.exists(temp_path):
                        try:
                            os.remove(temp_path)
                        except:
                            pass

        except Exception as e:
            flash(f'Ошибка при взаимодействии с API: {str(e)}', 'danger')
            return render_template('admin/simple_add_user.html', error=f'Ошибка при взаимодействии с API: {str(e)}')

    return render_template('admin/simple_add_user.html')

@app.route('/authorize', methods=['GET', 'POST'])
def voice_authorize():
    if request.method == 'POST':
        if 'audio_data' not in request.files:
            return jsonify({'error': 'No audio file uploaded'}), 400

        audio_file = request.files['audio_data']

        try:
            files = {'audio_data': (f"{uuid.uuid4()}.wav", audio_file)}
            response = requests.post(f"{API_URL}/api/authorize", files=files)
            result = response.json()

            if result.get('authorized', False):
                session['username'] = result.get('user', {}).get('name')
                session['user_id'] = result.get('user', {}).get('id')
                session['role'] = result.get('user', {}).get('role')
                return jsonify({'authorized': True, 'user': result.get('user', {})})
            else:
                return jsonify({'authorized': False, 'message': result.get('message', 'Authorization failed')})
        except Exception as e:
            return jsonify({'authorized': False, 'message': str(e)}), 500

    return render_template('authorize.html')

@app.route('/checkpoint')
@login_required
def checkpoint():
    """Страница режима контрольно-пропускного пункта"""
    return render_template('checkpoint.html')

@app.route('/api/voice/recent_events')
@login_required
def voice_recent_events():
    """Получение последних событий аутентификации"""
    try:
        # Запрос на API для получения последних логов аутентификации
        response = requests.get(
            f"{API_URL}/api/logs",
            params={
                "limit": 5,
                "event_type": "authorization_successful,authorization_attempt,spoofing_attempt"
            }
        )

        if response.status_code != 200:
            app.logger.error(f"Error getting logs: {response.text}")
            return jsonify({"events": []}), 500

        # Получение логов событий
        logs_data = response.json()
        logs = logs_data.get("logs", [])

        # Преобразование логов в формат событий аутентификации
        events = []
        for log in logs:
            # Тип события
            event_type = "success" if log.get("event_type") == "authorization_successful" else \
                "spoof" if log.get("event_type") == "spoofing_attempt" else "failure"

            # Детали события
            details = log.get("details", {})

            # Информация о пользователе
            user_info = None
            user_id = log.get("user_id")

            if user_id:
                try:
                    user_response = requests.get(f"{API_URL}/api/users/{user_id}")
                    if user_response.status_code == 200:
                        user = user_response.json().get("user", {})
                        user_info = {
                            "id": user_id,
                            "name": user.get("name", "Unknown user"),
                            "department": user.get("department", "-"),
                            "position": user.get("position", "-"),
                            "photo_url": f"/api-proxy/users/{user_id}/photo" if user.get("has_photo") else None
                        }
                except Exception as e:
                    app.logger.error(f"Error getting user info: {e}")

            # Формирование события
            event = {
                "id": log.get("id"),
                "timestamp": log.get("timestamp"),
                "type": event_type,
                "user": user_info,
                "authorized": event_type == "success",
                "spoofing_detected": event_type == "spoof",
                "match_score": details.get("match_score", 0),
                "similarity": details.get("similarity", 0)
            }

            events.append(event)

        return jsonify({"events": events})
    except Exception as e:
        app.logger.error(f"Error getting recent auth events: {e}")
        return jsonify({"events": []}), 500

@app.route('/api/voice/authenticate', methods=['POST'])
def voice_authenticate_proxy():
    """Прокси для голосовой аутентификации с расширенной диагностикой"""
    try:
        # 1. Проверка наличия файла
        if 'audio_data' not in request.files:
            app.logger.error("No audio file uploaded")
            return jsonify({
                'authorized': False,
                'message': 'Не загружен аудиофайл',
                'error_code': 'NO_FILE'
            }), 400

        audio_file = request.files['audio_data']

        # 2. Проверка имени файла
        if audio_file.filename == '':
            app.logger.error("Empty filename received")
            return jsonify({
                'authorized': False,
                'message': 'Некорректное имя файла',
                'error_code': 'INVALID_FILENAME'
            }), 400

        # 3. Создание директории для временных файлов
        shared_temp_dir = "/shared/temp"
        os.makedirs(shared_temp_dir, exist_ok=True)

        # 4. Генерация уникального имени файла
        file_uuid = str(uuid.uuid4())
        temp_file_path = os.path.join(shared_temp_dir, f"auth_{file_uuid}.wav")

        # 5. Сохранение файла с расширенной обработкой
        try:
            audio_file.save(temp_file_path)
        except PermissionError:
            app.logger.error(f"Permission denied when saving to {temp_file_path}")
            return jsonify({
                'authorized': False,
                'message': 'Ошибка сохранения файла',
                'error_code': 'SAVE_PERMISSION_DENIED'
            }), 500
        except Exception as save_error:
            app.logger.error(f"Error saving audio file: {save_error}")
            return jsonify({
                'authorized': False,
                'message': 'Не удалось сохранить аудиофайл',
                'error_code': 'FILE_SAVE_ERROR'
            }), 500

        # 6. Проверка файла
        if not os.path.exists(temp_file_path):
            app.logger.error(f"File was not saved: {temp_file_path}")
            return jsonify({
                'authorized': False,
                'message': 'Файл не был сохранен',
                'error_code': 'FILE_NOT_SAVED'
            }), 500

        # 7. Проверка размера файла
        file_size = os.path.getsize(temp_file_path)
        MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 МБ
        if file_size > MAX_FILE_SIZE:
            os.remove(temp_file_path)
            return jsonify({
                'authorized': False,
                'message': 'Превышен максимальный размер файла',
                'error_code': 'FILE_TOO_LARGE'
            }), 400

        # 8. Детальное логирование
        app.logger.info(f"Saved audio to: {temp_file_path}")
        app.logger.info(f"File size: {file_size} bytes")

        # 9. Отправка на API для авторизации
        try:
            with open(temp_file_path, 'rb') as f:
                files = {'audio_data': (os.path.basename(temp_file_path), f)}
                response = requests.post(
                    f"{API_URL}/api/authorize",
                    files=files,
                    timeout=30  # Таймаут для предотвращения зависания
                )

            # 10. Проверка ответа API
            if response.status_code != 200:
                app.logger.error(f"API returned non-200 status: {response.status_code}")
                return jsonify({
                    'authorized': False,
                    'message': 'Ошибка сервиса авторизации',
                    'error_code': 'AUTH_SERVICE_ERROR'
                }), 500

            # 11. Безопасное декодирование JSON
            try:
                result = response.json()
            except ValueError:
                app.logger.error(f"Invalid JSON from API: {response.text}")
                return jsonify({
                    'authorized': False,
                    'message': 'Некорректный ответ сервиса',
                    'error_code': 'INVALID_RESPONSE'
                }), 500

        except requests.RequestException as req_error:
            app.logger.error(f"Request to authorization service failed: {req_error}")
            return jsonify({
                'authorized': False,
                'message': 'Сервис авторизации недоступен',
                'error_code': 'AUTH_SERVICE_UNAVAILABLE'
            }), 500

        # 12. Удаление временного файла
        finally:
            try:
                if os.path.exists(temp_file_path):
                    os.remove(temp_file_path)
                    app.logger.info(f"Removed temporary file: {temp_file_path}")
            except Exception as cleanup_error:
                app.logger.warning(f"Failed to remove temp file: {cleanup_error}")

        # 13. Возврат результата
        return jsonify(result), 200

    except Exception as global_error:
        app.logger.error(f"Unexpected error in voice authentication: {global_error}", exc_info=True)
        return jsonify({
            'authorized': False,
            'message': 'Внутренняя ошибка сервера',
            'error_code': 'INTERNAL_SERVER_ERROR'
        }), 500

@app.route('/static/js/<path:filename>')
def serve_js(filename):
    """Обслуживание JavaScript файлов из директории static/js"""
    return send_from_directory('static/js', filename)

@app.route('/admin/users/<user_id>/delete', methods=['POST'])
@admin_required
def delete_user(user_id):
    """Удаление пользователя"""
    try:
        # Отправляем POST-запрос на удаление пользователя через API
        response = requests.post(f"{API_URL}/api/users/{user_id}/delete")

        result = response.json()

        if result.get('success', False):
            flash('User succesful deleted', 'success')
        else:
            # Получаем детали ошибки от API
            error_message = result.get('message', 'Failed to delete user')
            flash(f'Error deleting user: {error_message}', 'danger')

    except Exception as e:
        flash(f'Error deleting user: {str(e)}', 'danger')

    return redirect(url_for('manage_users'))



if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=os.environ.get('FLASK_DEBUG', '0') == '1')
    app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024