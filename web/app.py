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
from functools import wraps
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename

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
            error = 'Неверное имя пользователя или пароль'

    return render_template('login.html', error=error)

@app.route('/logout')
def logout():
    session.pop('username', None)
    return redirect(url_for('login'))


@app.route('/admin')
@admin_required
def admin_dashboard():
    """Главная страница администратора с актуальными данными"""
    try:
        # 1. Получаем системный статус
        status_response = requests.get(f"{API_URL}/api/system/status", timeout=10)
        system_status = {}
        if status_response.status_code == 200:
            system_status = status_response.json()

        # 2. Получаем список пользователей
        users_response = requests.get(f"{API_URL}/api/users", timeout=10)
        users_data = {}
        if users_response.status_code == 200:
            users_data = users_response.json()

        # 3. Получаем логи для статистики
        logs_response = requests.get(f"{API_URL}/api/logs", params={"limit": 100}, timeout=10)
        logs_data = {}
        if logs_response.status_code == 200:
            logs_data = logs_response.json()

        # 4. Обрабатываем данные для dashboard
        dashboard_data = process_dashboard_data(system_status, users_data, logs_data)

        return render_template('admin/dashboard.html',
                               username=session['username'],
                               **dashboard_data)

    except Exception as e:
        app.logger.error(f"Error loading dashboard data: {e}")
        # Возвращаем пустые данные в случае ошибки
        return render_template('admin/dashboard.html',
                               username=session['username'],
                               total_users=0,
                               active_users=0,
                               entries_today=0,
                               spoofing_attempts=0,
                               recent_events=[])


def process_dashboard_data(system_status, users_data, logs_data):
    """Обрабатывает данные для dashboard"""
    # Инициализация с безопасными значениями по умолчанию
    data = {
        'total_users': 0,
        'active_users': 0,
        'entries_today': 0,
        'spoofing_attempts': 0,
        'recent_events': []
    }

    # Обработка данных пользователей
    if system_status:
        data['total_users'] = system_status.get('users_count', 0)
        data['active_users'] = system_status.get('active_users_count', 0)

    # Обработка логов
    logs = logs_data.get('logs', []) if logs_data else []

    # Подсчет событий за сегодня
    today = datetime.date.today()
    entries_today = 0
    spoofing_attempts = 0

    for log in logs:
        try:
            # Парсим дату из лога
            log_date_str = log.get('timestamp', '')
            if isinstance(log_date_str, str):
                # Пробуем разные форматы даты
                log_date = None
                for fmt in ['%Y-%m-%dT%H:%M:%S.%f', '%Y-%m-%dT%H:%M:%S']:
                    try:
                        log_date = datetime.datetime.strptime(log_date_str.split('Z')[0], fmt).date()
                        break
                    except:
                        continue

                if log_date == today:
                    event_type = log.get('event_type', '')
                    if event_type == 'authorization_successful':
                        entries_today += 1
                    elif event_type == 'spoofing_attempt':
                        spoofing_attempts += 1
        except Exception as e:
            app.logger.warning(f"Error processing log date: {e}")
            continue

    data['entries_today'] = entries_today
    data['spoofing_attempts'] = spoofing_attempts

    # Обработка последних событий (берем первые 10)
    recent_events = []
    for log in logs[:10]:
        try:
            event = {
                'timestamp': format_timestamp(log.get('timestamp', '')),
                'type': get_event_display_name(log.get('event_type', '')),
                'user': log.get('user_name', 'Unknown'),
                'status': 'Success' if log.get('success', False) else 'Failed'
            }
            recent_events.append(event)
        except Exception as e:
            app.logger.warning(f"Error processing event: {e}")
            continue

    data['recent_events'] = recent_events

    return data


def format_timestamp(timestamp_str):
    """Форматирует временную метку для отображения"""
    if not timestamp_str:
        return 'Unknown'

    try:
        # Пробуем разные форматы
        for fmt in ['%Y-%m-%dT%H:%M:%S.%f', '%Y-%m-%dT%H:%M:%S']:
            try:
                dt = datetime.datetime.strptime(timestamp_str.split('Z')[0], fmt)
                return dt.strftime('%Y-%m-%d %H:%M:%S')
            except:
                continue
        return timestamp_str
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
        'model_deployed': 'Model Deployed'
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
                error = f"Ошибка при создании пользователя: {error_detail}"
                return render_template('admin/add_user.html', error=error)
        except Exception as e:
            error = f"Ошибка соединения с API: {str(e)}"
            return render_template('admin/add_user.html', error=error)

    return render_template('admin/add_user.html')


@app.route('/api-proxy/<path:subpath>')
@admin_required
def api_proxy(subpath):
    """
    Прокси для API запросов к серверу API.
    Используется для получения файлов, таких как фотографии пользователей.
    """
    try:
        # Создаем полный URL для запроса к API
        api_url = f"{API_URL}/api/{subpath}"

        # Отправляем запрос к API с теми же параметрами, что и оригинальный запрос
        response = requests.get(
            api_url,
            params=request.args,
            stream=True,
            headers={key: value for key, value in request.headers if key != 'Host'}
        )

        # Создаем объект Response с данными из API
        flask_response = Response(
            response=response.raw.read(),
            status=response.status_code
        )

        # Копируем заголовки из ответа API
        for key, value in response.headers.items():
            if key.lower() not in ('content-length', 'connection', 'content-encoding'):
                flask_response.headers[key] = value

        return flask_response
    except Exception as e:
        app.logger.error(f"Error in API proxy: {str(e)}")
        return Response(f"Error connecting to API server: {str(e)}", status=500)

# web/app.py - добавьте маршрут для загрузки фото пользователя
@app.route('/admin/users/<user_id>/upload_photo', methods=['POST'])
@admin_required
def upload_user_photo(user_id):
    """Загрузка фотографии пользователя"""
    try:
        if 'photo' not in request.files:
            flash('Не выбрано фото', 'danger')
            return redirect(url_for('enroll_user_voice', user_id=user_id))

        photo = request.files['photo']

        if photo.filename == '':
            flash('Не выбрано фото', 'danger')
            return redirect(url_for('enroll_user_voice', user_id=user_id))

        # Проверка типа файла
        allowed_extensions = {'jpg', 'jpeg', 'png'}
        if not photo.filename.lower().split('.')[-1] in allowed_extensions:
            flash('Недопустимый формат файла. Разрешены только JPG и PNG', 'danger')
            return redirect(url_for('enroll_user_voice', user_id=user_id))

        # Получение информации о пользователе
        response = requests.get(f"{API_URL}/api/users/{user_id}")

        if response.status_code != 200:
            flash('Пользователь не найден', 'danger')
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
            flash('Фото успешно загружено', 'success')
        else:
            flash(f'Ошибка при загрузке фото: {upload_response.json().get("detail", "Неизвестная ошибка")}', 'danger')

        # Удаление временного файла
        if os.path.exists(photo_path):
            os.remove(photo_path)

        return redirect(url_for('enroll_user_voice', user_id=user_id))
    except Exception as e:
        flash(f'Ошибка: {str(e)}', 'danger')
        return redirect(url_for('enroll_user_voice', user_id=user_id))


@app.route('/admin/users/<user_id>/enroll', methods=['GET', 'POST'])
@admin_required
def enroll_user_voice(user_id):
    # Получение информации о пользователе
    try:
        response = requests.get(f"{API_URL}/api/users/{user_id}")
        user = response.json().get('user', {})
    except:
        user = {'id': user_id, 'name': 'Неизвестный пользователь'}

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
    """Получает информацию о Docker контейнерах (заглушка)"""
    # В реальной реализации здесь бы был запрос к Docker API
    # Пока возвращаем заглушку с базовой информацией
    containers = [
        {
            'name': 'voice-auth-api',
            'status': 'Running',
            'cpu': '2.5%',
            'memory': '128 MB',
            'network': '1.2 KB/s',
            'uptime': '2d 14h'
        },
        {
            'name': 'voice-auth-ml',
            'status': 'Running',
            'cpu': '15.3%',
            'memory': '512 MB',
            'network': '5.7 KB/s',
            'uptime': '2d 14h'
        },
        {
            'name': 'voice-auth-db',
            'status': 'Running',
            'cpu': '1.1%',
            'memory': '64 MB',
            'network': '0.8 KB/s',
            'uptime': '2d 14h'
        },
        {
            'name': 'voice-auth-web',
            'status': 'Running',
            'cpu': '0.8%',
            'memory': '32 MB',
            'network': '2.1 KB/s',
            'uptime': '2d 14h'
        }
    ]
    return containers


def get_default_status():
    """Возвращает статус по умолчанию в случае ошибки"""
    return {
        'api_status': 'unknown',
        'ml_status': 'unknown',
        'db_status': 'unknown',
        'users_count': 0,
        'active_users_count': 0,
        'device': 'Unknown',
        'api_version': 'Unknown',
        'storage': {
            'audio_used': '0 MB',
            'audio_total': '1 GB',
            'audio_percent': 0,
            'db_used': '0 MB',
            'db_total': '1 GB',
            'db_percent': 0,
            'ml_used': '0 MB',
            'ml_total': '1 GB',
            'ml_percent': 0
        },
        'containers': [],
        'backups': [],
        'backup_schedule': 'Not configured'
    }

@app.route('/admin/auth-monitor')
@admin_required
def auth_monitor():
    """Страница мониторинга голосовой аутентификации"""
    return render_template('admin/auth_monitor.html')


@app.route('/admin/system/reinitialize', methods=['POST'])
@admin_required
def reinitialize_system_web():
    """Переинициализация системы через веб-интерфейс"""
    try:
        # Вызов API для переинициализации системы
        response = requests.post(f"{API_URL}/api/system/reinitialize")
        
        if response.status_code == 200:
            result = response.json()
            
            if result.get('success', False):
                # Система успешно переинициализирована
                flash('Система успешно переинициализирована', 'success')
            else:
                # Произошла ошибка при переинициализации
                error_message = result.get('error', 'Неизвестная ошибка')
                flash(f'Ошибка при переинициализации системы: {error_message}', 'danger')
        else:
            # Ошибка API запроса
            flash(f'Ошибка API запроса: {response.status_code}', 'danger')
        
        # Перенаправляем на страницу статуса системы
        return redirect(url_for('system_status'))
        
    except Exception as e:
        flash(f'Ошибка при переинициализации системы: {str(e)}', 'danger')
        return redirect(url_for('system_status'))

# web/app.py - добавьте новый маршрут
@app.route('/admin/system/filesystem')
@admin_required
def filesystem_diagnosis():
    """Страница диагностики файловой системы"""
    try:
        response = requests.get(f"{API_URL}/api/system/filesystem")
        fs_data = response.json() if response.status_code == 200 else {"error": "Failed to get filesystem data"}

        return render_template('admin/filesystem_diagnosis.html', fs_data=fs_data)
    except Exception as e:
        flash(f'Ошибка при получении данных файловой системы: {str(e)}', 'danger')
        return redirect(url_for('system_status'))


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
# НОВЫЕ МАРШРУТЫ ДЛЯ УПРОЩЕННОГО ДОБАВЛЕНИЯ ПОЛЬЗОВАТЕЛЕЙ И ПЕРЕНОСА СИСТЕМЫ

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
                    status_message = f"Пользователь {name} успешно создан с {upload_result.get('processed_files', 0)} аудиофайлами"

                    if upload_result.get("activation_status", False):
                        status_message += " и активирован"
                    else:
                        status_message += " (не активирован)"

                    flash(status_message, 'success')
                    return render_template('admin/simple_add_user.html', success=status_message)
                else:
                    error_message = upload_result.get("message", "Неизвестная ошибка при загрузке аудиофайлов")
                    flash(error_message, 'danger')
                    return render_template('admin/simple_add_user.html', error=error_message)

            except Exception as e:
                flash(f'Ошибка при загрузке аудиофайлов: {str(e)}', 'danger')
                return render_template('admin/simple_add_user.html', error=f'Ошибка при загрузке аудиофайлов: {str(e)}')

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

@app.route('/admin/users/batch_import/zip', methods=['POST'])
@admin_required
def batch_import_users_from_zip():
    """Обработка пакетного импорта из ZIP-архива"""
    if 'zip_file' not in request.files:
        flash('Нет выбранного ZIP-архива', 'danger')
        return redirect(url_for('batch_import_view'))

    zip_file = request.files['zip_file']

    if not zip_file or not zip_file.filename or not zip_file.filename.lower().endswith('.zip'):
        flash('Неверный формат файла. Требуется ZIP-архив', 'danger')
        return redirect(url_for('batch_import_view'))

    # Создаем временную директорию для обработки
    temp_dir = tempfile.mkdtemp(dir=UPLOAD_FOLDER)
    zip_path = os.path.join(temp_dir, secure_filename(zip_file.filename))

    try:
        # Сохраняем архив
        zip_file.save(zip_path)

        # Распаковываем архив
        extract_dir = os.path.join(temp_dir, 'extract')
        os.makedirs(extract_dir, exist_ok=True)

        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)

        # Находим директории пользователей
        user_dirs = [d for d in os.listdir(extract_dir)
                     if os.path.isdir(os.path.join(extract_dir, d)) and d != '__MACOSX']

        if not user_dirs:
            flash('В архиве не найдены директории пользователей', 'danger')
            return redirect(url_for('batch_import_view'))

        # Обрабатываем каждого пользователя
        success_count = 0
        total_count = len(user_dirs)

        for user_dir_name in user_dirs:
            user_dir = os.path.join(extract_dir, user_dir_name)

            # Ищем WAV-файлы
            wav_files = [os.path.join(user_dir, f) for f in os.listdir(user_dir)
                         if os.path.isfile(os.path.join(user_dir, f)) and f.lower().endswith('.wav')]

            if len(wav_files) < 5:
                continue

            try:
                # Создаем пользователя
                display_name = user_dir_name.replace('_', ' ')
                response = requests.post(f"{API_URL}/api/simple/users", data={"name": display_name})
                response.raise_for_status()

                result = response.json()
                if not result.get("success", False):
                    continue

                user_id = result.get("user_id")

                # Загружаем аудиофайлы
                files = []
                for file_path in wav_files:
                    if os.path.exists(file_path):
                        files.append(('audio_files', (os.path.basename(file_path), open(file_path, 'rb'), 'audio/wav')))

                # Отправляем запрос на загрузку аудиофайлов
                try:
                    upload_response = requests.post(
                        f"{API_URL}/api/simple/upload_voice_batch",
                        data={"user_id": user_id},
                        files=files
                    )
                    upload_response.raise_for_status()

                    # Закрываем все открытые файлы
                    for _, file_tuple, _, _ in files:
                        file_tuple.close()

                    upload_result = upload_response.json()

                    if upload_result.get("success", False):
                        success_count += 1

                except Exception as e:
                    # Закрываем все открытые файлы при ошибке
                    for _, file_tuple, _, _ in files:
                        try:
                            file_tuple.close()
                        except:
                            pass

            except Exception as e:
                continue

        flash(f'Успешно импортировано {success_count} из {total_count} пользователей', 'success' if success_count > 0 else 'warning')
        return redirect(url_for('manage_users'))

    except Exception as e:
        flash(f'Ошибка при импорте пользователей из архива: {str(e)}', 'danger')
        return redirect(url_for('batch_import_view'))

    finally:
        # Удаляем временную директорию
        shutil.rmtree(temp_dir, ignore_errors=True)

@app.route('/admin/system/export', methods=['POST'])
@admin_required
def export_system():
    """Экспорт системы в ZIP-архив"""
    # Получение параметров
    include_audio = request.form.get('include_audio', 'on') == 'on'
    include_logs = request.form.get('include_logs', 'on') == 'on'

    try:
        # Вызов API для экспорта
        response = requests.get(f"{API_URL}/api/system/export")
        response.raise_for_status()

        result = response.json()

        if not result.get("success", False):
            return jsonify({"success": False, "message": result.get("message", "Ошибка экспорта")})

        export_filename = result.get("export_file")
        download_url = result.get("download_url")

        # Загружаем файл с API
        file_response = requests.get(f"{API_URL}{download_url}", stream=True)
        file_response.raise_for_status()

        # Сохраняем в директорию экспорта
        export_file_path = os.path.join(EXPORT_FOLDER, export_filename)
        with open(export_file_path, 'wb') as f:
            for chunk in file_response.iter_content(chunk_size=8192):
                f.write(chunk)

        return jsonify({
            "success": True,
            "download_url": url_for('download_export', filename=export_filename),
            "message": "Экспорт успешно выполнен"
        })

    except Exception as e:
        return jsonify({"success": False, "message": f"Ошибка при экспорте: {str(e)}"})

@app.route('/admin/downloads/<filename>')
@admin_required
def download_export(filename):
    """Скачивание файла экспорта"""
    return send_from_directory(EXPORT_FOLDER, filename, as_attachment=True)


@app.route('/admin/users/add_options')
@admin_required
def user_add_options():
    """Страница с опциями добавления пользователей"""
    return render_template('admin/user_add_options.html')

@app.route('/admin/system/import', methods=['POST'])
@admin_required
def import_system():
    """Импорт системы из ZIP-архива"""
    if 'import_file' not in request.files:
        return jsonify({"success": False, "message": "Файл не выбран"})

    import_file = request.files['import_file']

    if not import_file or not import_file.filename or not import_file.filename.lower().endswith('.zip'):
        return jsonify({"success": False, "message": "Неверный формат файла. Требуется ZIP-архив"})

    # Проверка подтверждения
    if request.form.get('confirm_import') != 'on':
        return jsonify({"success": False, "message": "Необходимо подтвердить импорт"})

    try:
        # Сохранение файла во временную директорию
        temp_file = os.path.join(UPLOAD_FOLDER, secure_filename(import_file.filename))
        import_file.save(temp_file)

        # Отправка файла в API
        with open(temp_file, 'rb') as f:
            response = requests.post(
                f"{API_URL}/api/system/import",
                files={"import_file": (os.path.basename(temp_file), f)}
            )

        response.raise_for_status()

        # Удаление временного файла
        if os.path.exists(temp_file):
            os.remove(temp_file)

        result = response.json()

        if not result.get("success", False):
            return jsonify({"success": False, "message": result.get("message", "Ошибка импорта")})

        return jsonify({
            "success": True,
            "message": "Импорт успешно выполнен"
        })

    except Exception as e:
        # Попытка удаления временного файла при ошибке
        if 'temp_file' in locals() and os.path.exists(temp_file):
            os.remove(temp_file)

        return jsonify({"success": False, "message": f"Ошибка при импорте: {str(e)}"})

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

# Добавьте этот маршрут в web/app.py
@app.route('/checkpoint')
@login_required
def checkpoint():
    """Страница режима контрольно-пропускного пункта"""
    return render_template('checkpoint.html')

# Прокси-маршруты для API (чтобы избежать проблем с CORS)
@app.route('/api/kpp/start', methods=['POST'])
@login_required
def proxy_kpp_start():
    """Прокси для запуска КПП"""
    try:
        response = requests.post(f"{API_URL}/api/kpp/start")
        return jsonify(response.json())
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/api/kpp/stop', methods=['POST'])
@login_required
def proxy_kpp_stop():
    """Прокси для остановки КПП"""
    try:
        response = requests.post(f"{API_URL}/api/kpp/stop")
        return jsonify(response.json())
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/api/kpp/status')
@login_required
def proxy_kpp_status():
    """Прокси для получения статуса КПП"""
    try:
        response = requests.get(f"{API_URL}/api/kpp/status")
        return jsonify(response.json())
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route('/api/voice/kpp_mode', methods=['POST'])
@login_required
def proxy_voice_kpp_mode():
    """Прокси для управления режимом КПП"""
    try:
        data = request.get_json()
        response = requests.post(f"{API_URL}/api/kpp/start" if data.get('enabled', False) else f"{API_URL}/api/kpp/stop", json=data)
        return jsonify(response.json()), response.status_code
    except Exception as e:
        app.logger.error(f"Error in KPP mode proxy: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500


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
                            "name": user.get("name", "Неизвестный пользователь"),
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
                    timeout=10  # Таймаут для предотвращения зависания
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

# Добавьте этот маршрут в web/app.py

@app.post("/system/reinitialize")
async def reinitialize_system():
    """Полная переинициализация системы"""
    try:
        # Перезагрузка эмбеддингов
        emb_success = force_reload_embeddings()

        # Сброс порогов
        global adaptive_threshold, SPOOFING_THRESHOLD
        adaptive_threshold = 0.5  # Установите низкий порог для тестирования
        SPOOFING_THRESHOLD = 0.7  # Высокий порог для спуфинга

        return {
            "success": True,
            "embeddings_reloaded": emb_success,
            "user_count": len(user_embeddings),
            "adaptive_threshold": adaptive_threshold,
            "spoofing_threshold": SPOOFING_THRESHOLD
        }
    except Exception as e:
        logger.error(f"Error reinitializing system: {e}")
        return {
            "success": False,
            "error": str(e)
        }

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
            flash('Пользователь успешно удален', 'success')
        else:
            # Получаем детали ошибки от API
            error_message = result.get('message', 'Не удалось удалить пользователя')
            flash(f'Ошибка при удалении пользователя: {error_message}', 'danger')

    except Exception as e:
        flash(f'Ошибка при удалении пользователя: {str(e)}', 'danger')

    return redirect(url_for('manage_users'))



if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=os.environ.get('FLASK_DEBUG', '0') == '1')
    app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024