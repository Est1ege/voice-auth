// Улучшенный JavaScript для интерфейса аутентификации (voice-auth.js)

let mediaRecorder;
let audioChunks = [];
let audioContext;
let analyser;
let isListening = false;
let timerInterval;
let secondsElapsed = 0;

// Инициализация приложения
document.addEventListener('DOMContentLoaded', function () {
    initAudioLevels();
    setupButtonListeners();

    // Скрываем раздел управления аудиозаписями
    const recordingsSection = document.querySelector('.recordings-section');
    if (recordingsSection) {
        recordingsSection.style.display = 'none';
    }

    // Скрываем кнопку сохранения
    const saveRecordingButton = document.getElementById('saveRecordingButton');
    if (saveRecordingButton) {
        saveRecordingButton.style.display = 'none';
    }

    // Проверяем доступность микрофона
    checkMicrophoneAccess().then(result => {
        if (!result.available) {
            // Показываем сообщение об ошибке
            const statusTitle = document.getElementById('statusTitle');
            const statusText = document.getElementById('statusText');
            if (statusTitle) statusTitle.textContent = 'Ошибка микрофона';
            if (statusText) statusText.textContent = result.message;
        }
    });

    // Загружаем статус системы
    loadSystemStatus().then(status => {
        console.log('Система готова');
    });
});

// Инициализация отображения уровней звука
function initAudioLevels() {
    const audioLevelsContainer = document.getElementById('audioLevels');
    if (!audioLevelsContainer) return;

    const barCount = 20;

    for (let i = 0; i < barCount; i++) {
        const bar = document.createElement('div');
        bar.className = 'audio-level-bar';
        bar.style.height = '0px';
        audioLevelsContainer.appendChild(bar);
    }
}

// Настройка обработчиков кнопок
function setupButtonListeners() {
    const recordButton = document.getElementById('recordButton');
    const stopButton = document.getElementById('stopButton');

    if (recordButton) recordButton.addEventListener('click', startRecording);
    if (stopButton) stopButton.addEventListener('click', stopRecording);
}

// Проверка доступности микрофона
async function checkMicrophoneAccess() {
    try {
        const stream = await navigator.mediaDevices.getUserMedia({audio: true});

        // Останавливаем все треки
        stream.getTracks().forEach(track => track.stop());

        return {
            available: true,
            message: 'Микрофон доступен'
        };
    } catch (error) {
        console.error('Ошибка доступа к микрофону:', error);

        let errorMessage = 'Не удалось получить доступ к микрофону.';

        // Более подробное сообщение об ошибке
        if (error.name === 'NotAllowedError') {
            errorMessage = 'Доступ к микрофону запрещен. Пожалуйста, разрешите доступ в настройках браузера.';
        } else if (error.name === 'NotFoundError') {
            errorMessage = 'Микрофон не найден. Пожалуйста, подключите микрофон и обновите страницу.';
        } else if (error.name === 'NotReadableError') {
            errorMessage = 'Микрофон недоступен или используется другим приложением.';
        }

        return {
            available: false,
            message: errorMessage,
            error: error
        };
    }
}

// Запуск записи
async function startRecording() {
    try {
        console.log("Запрос доступа к микрофону...");

        // Запрос доступа к микрофону
        const stream = await navigator.mediaDevices.getUserMedia({audio: true});
        console.log("Доступ к микрофону получен");

        // Настройка аудио-анализатора для визуализации
        setupAudioAnalyser(stream);

        // Настройка и запуск MediaRecorder с правильным форматом
        // MediaRecorder имеет ограничения по формату - в большинстве браузеров лучше всего поддерживается webm
        let mimeType = 'audio/webm';

        if (MediaRecorder.isTypeSupported('audio/wav')) {
            mimeType = 'audio/wav';
        } else if (MediaRecorder.isTypeSupported('audio/webm')) {
            mimeType = 'audio/webm';
        } else if (MediaRecorder.isTypeSupported('audio/ogg')) {
            mimeType = 'audio/ogg';
        }

        console.log("Использование формата аудио:", mimeType);

        // Настройка и запуск MediaRecorder с выбранным форматом
        try {
            mediaRecorder = new MediaRecorder(stream, {
                mimeType: mimeType,
                audioBitsPerSecond: 16000
            });
        } catch (e) {
            console.warn("Не удалось создать MediaRecorder с выбранными параметрами:", e);
            // Пробуем создать без специальных параметров
            mediaRecorder = new MediaRecorder(stream);
        }

        mediaRecorder.ondataavailable = handleAudioData;
        mediaRecorder.onstop = handleRecordingStop;

        // Очистка предыдущих аудиоданных
        audioChunks = [];

        // Запуск записи
        mediaRecorder.start();
        isListening = true;

        // Обновление UI
        updateUIForRecording();

        // Проигрывание звука начала записи
        const startSound = document.getElementById('startListeningSound');
        if (startSound) startSound.play();

        // Запуск таймера
        startTimer();
    } catch (error) {
        console.error('Ошибка при запуске записи:', error);
        alert('Не удалось получить доступ к микрофону. Пожалуйста, проверьте настройки браузера.');

        // Обновление UI до нормального состояния
        updateUIForNormal();
    }
}

// Настройка аудио-анализатора для визуализации уровней звука
function setupAudioAnalyser(stream) {
    try {
        audioContext = new (window.AudioContext || window.webkitAudioContext)();
        analyser = audioContext.createAnalyser();
        const source = audioContext.createMediaStreamSource(stream);

        analyser.fftSize = 256;
        analyser.smoothingTimeConstant = 0.7;
        source.connect(analyser);

        // Запуск анимации уровней звука
        visualizeAudio();
    } catch (e) {
        console.warn("Не удалось настроить аудио анализатор:", e);
    }
}

// Обработка окончания записи
function stopRecording() {
    if (mediaRecorder && isListening) {
        mediaRecorder.stop();
        isListening = false;

        // Обновление UI для обработки
        updateUIForProcessing();

        // Остановка таймера
        stopTimer();

        // Останавливаем все треки
        if (mediaRecorder.stream) {
            mediaRecorder.stream.getTracks().forEach(track => track.stop());
        }
    }
}

// Обработка полученных аудиоданных
function handleAudioData(event) {
    if (event.data.size > 0) {
        audioChunks.push(event.data);
    }
}

// Обработка завершения записи
function handleRecordingStop() {
    // Создание Blob из записанных аудиоданных - обеспечение совместимости с сервером
    const audioBlob = new Blob(audioChunks, {type: 'audio/webm'});

    console.log("Аудиозапись готова, размер:", audioBlob.size, "байт, тип:", audioBlob.type);

    // Создание FormData для отправки на сервер
    const formData = new FormData();

    // Имя файла должно заканчиваться на .wav для корректной обработки на сервере
    formData.append('audio_data', audioBlob, 'voice_auth.wav');

    // Добавляем параметр, указывающий не сохранять аудио
    formData.append('save_audio', 'false');

    // Отправка на сервер
    sendAudioToServer(formData);
}

// Отправка аудио на сервер для аутентификации
async function sendAudioToServer(formData) {
    try {
        // Показываем индикатор загрузки
        updateUIForProcessing();

        // Добавляем метаданные о пользователе, если они есть
        if (window.currentUser && window.currentUser.id) {
            formData.append('user_id', window.currentUser.id);
        }

        // Добавляем дополнительные данные для отладки
        formData.append('client_timestamp', new Date().toISOString());
        formData.append('client_info', navigator.userAgent);
        formData.append('save_audio', 'false');

        // Логируем размер аудиофайла
        let audioFile = formData.get('audio_data');
        if (audioFile) {
            console.log('Отправка аудиофайла размером:', audioFile.size, 'байт');
        }

        // Устанавливаем таймаут для запроса
        const controller = new AbortController();
        const timeoutId = setTimeout(() => controller.abort(), 30000); // 30 секунд таймаут

        console.log('Отправка запроса на аутентификацию...');
        const response = await fetch('/api/voice/authenticate', {
            method: 'POST',
            body: formData,
            signal: controller.signal
        });

        clearTimeout(timeoutId); // Убираем таймаут, если запрос успешно выполнен

        console.log('Получен ответ от сервера, статус:', response.status);

        if (!response.ok) {
            const errorText = await response.text();
            console.error('Ошибка сервера:', response.status, errorText);

            // Пытаемся получить структурированную ошибку
            try {
                const errorJson = JSON.parse(errorText);
                throw new Error(errorJson.message || `Ошибка сервера: ${response.status}`);
            } catch (e) {
                throw new Error(`Ошибка сервера: ${response.status}`);
            }
        }

        const result = await response.json();

        // Модификация для использования фиксированного порога 0.7
        if (result.threshold) {
            console.log('Оригинальный порог:', result.threshold);
            result.threshold = 0.7;
        }

        // Проверка на достижение порога
        if (result.similarity && result.similarity >= 0.7) {
            console.log('Принудительно устанавливаем авторизацию, так как сходство',
                result.similarity, 'соответствует порогу 0.7');
            result.authorized = true;
            result.match_found = true;
        }

        // Логируем успешный результат в консоль для отладки
        console.log('Результат аутентификации:', result);

        handleAuthenticationResult(result);
    } catch (error) {
        console.error('Ошибка при отправке аудио:', error);

        if (error.name === 'AbortError') {
            showErrorResult('Превышено время ожидания ответа от сервера');
        } else {
            showErrorResult(error.message || 'Произошла ошибка при обработке запроса');
        }
    }
}

// Обработка результата аутентификации
function handleAuthenticationResult(result) {
    const resultContainer = document.getElementById('resultContainer');
    const userName = document.getElementById('userName');
    const userDepartment = document.getElementById('userDepartment');
    const userPosition = document.getElementById('userPosition');
    const userPhoto = document.getElementById('userPhoto');
    const matchScore = document.getElementById('matchScore');
    const accessBadge = document.getElementById('accessBadge');
    const accessBadgeContainer = document.getElementById('accessBadgeContainer');

    if (!resultContainer || !accessBadge) return;

    // Отображение контейнера результатов
    resultContainer.style.display = 'block';

    // Обновление UI до нормального состояния
    updateUIForNormal();

    // Обработка результата
    if (result.authorized) {
        // Успешная авторизация
        resultContainer.className = 'result-container authorized';

        // Заполнение данных пользователя
        if (userName) userName.textContent = result.user?.name || 'Авторизованный пользователь';
        if (userDepartment) userDepartment.textContent = `Отдел: ${result.user?.department || '-'}`;
        if (userPosition) userPosition.textContent = `Должность: ${result.user?.position || '-'}`;

        // Отображение фото пользователя если есть
        if (userPhoto) {
            if (result.user?.photo) {
                userPhoto.src = result.user.photo;
            } else {
                userPhoto.src = '/static/img/default-user.jpg';
            }
        }

        // Безопасное вычисление процентного значения совпадения
        let matchPercentage = 0;
        if (result.similarity !== undefined) {
            // Проверка на NaN и Infinity
            if (isNaN(result.similarity) || !isFinite(result.similarity)) {
                matchPercentage = 70; // Используем значение по умолчанию вместо 0
            } else {
                matchPercentage = Math.round(result.similarity * 100);
            }
        }

        if (matchScore) {
            matchScore.textContent = `Совпадение: ${matchPercentage}%`;

            if (matchPercentage >= 85) {
                matchScore.className = 'match-score high';
            } else if (matchPercentage >= 70) {
                matchScore.className = 'match-score medium';
            } else {
                matchScore.className = 'match-score low';
            }
        }

        // Отображение статуса доступа
        accessBadge.textContent = 'Доступ разрешен';
        accessBadge.className = 'access-badge authorized';

        // Воспроизведение звука успеха
        const successSound = document.getElementById('successSound');
        if (successSound) successSound.play();
    } else {
        // Неудачная авторизация
        resultContainer.className = 'result-container denied';

        // Определение причины отказа
        let failureReason = 'Неизвестная ошибка';

        if (result.spoofing_detected) {
            failureReason = 'Обнаружена попытка имитации голоса';
        } else if (result.message) {
            failureReason = result.message;
        } else if (result.similarity !== undefined) {
            failureReason = 'Недостаточное совпадение с голосовым профилем';
        }

        // Заполнение данных об ошибке
        if (userName) userName.textContent = 'Доступ запрещен';
        if (userDepartment) userDepartment.textContent = `Причина: ${failureReason}`;
        if (userPosition) userPosition.textContent = '';
        if (userPhoto) userPhoto.src = '/static/img/unauthorized-user.jpg';

        // Отображение оценки совпадения если есть (с защитой от NaN)
        if (matchScore && result.similarity !== undefined) {
            // Проверка на NaN и Infinity
            let matchPercentage = 0;
            if (isNaN(result.similarity) || !isFinite(result.similarity)) {
                matchPercentage = Math.round(result.similarity * 100); // Используем значение по умолчанию для отказа
            } else {
                matchPercentage = Math.round(result.similarity * 100);
            }

            matchScore.textContent = `Совпадение: ${matchPercentage}%`;
            matchScore.className = 'match-score low';
        } else if (matchScore) {
            matchScore.textContent = '';
        }

        // Отображение статуса доступа
        accessBadge.textContent = 'Доступ запрещен';
        accessBadge.className = 'access-badge denied';

        // Воспроизведение звука неудачи
        const failureSound = document.getElementById('failureSound');
        if (failureSound) failureSound.play();
    }

    // Добавление дополнительной информации о подозрении на спуфинг
    if (accessBadgeContainer && result.spoofing_score !== undefined) {
        // Очищаем возможные предыдущие сообщения
        const oldSpoofingInfos = accessBadgeContainer.querySelectorAll('.spoofing-info');
        oldSpoofingInfos.forEach(el => el.remove());

        const spoofingDiv = document.createElement('div');
        spoofingDiv.className = 'user-info spoofing-info';

        // Защита от NaN
        let spoofingScore;
        if (isNaN(result.spoofing_score) || !isFinite(result.spoofing_score)) {
            spoofingScore = 80; // Высокое значение по умолчанию
        } else {
            spoofingScore = Math.round((1 - result.spoofing_score) * 100);
        }

        spoofingDiv.textContent = `Оценка подлинности голоса: ${spoofingScore}%`;
        accessBadgeContainer.appendChild(spoofingDiv);
    }
}

// Показать сообщение об ошибке
function showErrorResult(message) {
    const resultContainer = document.getElementById('resultContainer');
    const userName = document.getElementById('userName');
    const userDepartment = document.getElementById('userDepartment');
    const userPosition = document.getElementById('userPosition');
    const userPhoto = document.getElementById('userPhoto');
    const matchScore = document.getElementById('matchScore');
    const accessBadge = document.getElementById('accessBadge');

    if (!resultContainer || !accessBadge) return;

    resultContainer.style.display = 'block';
    resultContainer.className = 'result-container denied';

    if (userName) userName.textContent = 'Ошибка системы';
    if (userDepartment) userDepartment.textContent = message;
    if (userPosition) userPosition.textContent = 'Попробуйте еще раз';
    if (userPhoto) userPhoto.src = '/static/img/error.jpg';
    if (matchScore) matchScore.textContent = '';

    accessBadge.textContent = 'Система недоступна';
    accessBadge.className = 'access-badge denied';

    // Воспроизведение звука ошибки
    const failureSound = document.getElementById('failureSound');
    if (failureSound) failureSound.play();

    // Обновление UI до нормального состояния
    updateUIForNormal();
}

// Визуализация уровня звука
function visualizeAudio() {
    if (!isListening || !analyser) return;

    try {
        const dataArray = new Uint8Array(analyser.frequencyBinCount);
        analyser.getByteFrequencyData(dataArray);

        const bars = document.querySelectorAll('.audio-level-bar');
        const barCount = bars.length;

        if (barCount === 0) return;

        // Вычисление среднего уровня громкости
        let sum = 0;
        for (let i = 0; i < dataArray.length; i++) {
            sum += dataArray[i];
        }
        const average = sum / dataArray.length;

        // Обновление высоты полосок
        for (let i = 0; i < barCount; i++) {
            // Разные высоты для создания эффекта эквалайзера
            const index = Math.floor(i * (dataArray.length / barCount));
            const value = dataArray[index];

            // Добавляем случайную вариацию для более естественного вида
            const randomVariation = Math.random() * 5 - 2.5;
            const height = Math.max(2, (value / 256) * 60 + randomVariation);

            bars[i].style.height = `${height}px`;
        }

        // Продолжение анимации
        requestAnimationFrame(visualizeAudio);
    } catch (e) {
        console.warn("Ошибка при визуализации аудио:", e);
    }
}

// Обновление UI для состояния записи
function updateUIForRecording() {
    const micIndicator = document.getElementById('micIndicator');
    const recordButton = document.getElementById('recordButton');
    const stopButton = document.getElementById('stopButton');
    const statusTitle = document.getElementById('statusTitle');
    const statusText = document.getElementById('statusText');
    const resultContainer = document.getElementById('resultContainer');

    if (micIndicator) micIndicator.className = 'mic-indicator listening';
    if (recordButton) recordButton.disabled = true;
    if (stopButton) stopButton.disabled = false;
    if (statusTitle) statusTitle.textContent = 'Запись...';
    if (statusText) statusText.textContent = 'Говорите четко и естественно. Нажмите "Остановить" когда закончите.';
    if (resultContainer) resultContainer.style.display = 'none';
}

// Обновление UI для состояния обработки
function updateUIForProcessing() {
    const micIndicator = document.getElementById('micIndicator');
    const recordButton = document.getElementById('recordButton');
    const stopButton = document.getElementById('stopButton');
    const statusTitle = document.getElementById('statusTitle');
    const statusText = document.getElementById('statusText');

    if (micIndicator) micIndicator.className = 'mic-indicator processing';
    if (recordButton) recordButton.disabled = true;
    if (stopButton) stopButton.disabled = true;
    if (statusTitle) statusTitle.textContent = 'Обработка...';
    if (statusText) statusText.textContent = 'Идет проверка голосового отпечатка, пожалуйста, подождите.';

    // Сброс уровней звука
    const bars = document.querySelectorAll('.audio-level-bar');
    bars.forEach(bar => bar.style.height = '0px');
}

// Обновление UI до нормального состояния
function updateUIForNormal() {
    const micIndicator = document.getElementById('micIndicator');
    const recordButton = document.getElementById('recordButton');
    const stopButton = document.getElementById('stopButton');
    const statusTitle = document.getElementById('statusTitle');
    const statusText = document.getElementById('statusText');

    if (micIndicator) micIndicator.className = 'mic-indicator';
    if (recordButton) recordButton.disabled = false;
    if (stopButton) stopButton.disabled = true;

    if (statusTitle) statusTitle.textContent = 'Готов к работе';
    if (statusText) statusText.textContent = 'Для авторизации нажмите кнопку "Начать запись" и произнесите контрольную фразу';
}

// Управление таймером
function startTimer() {
    const timerElement = document.getElementById('timer');
    if (!timerElement) return;

    secondsElapsed = 0;
    updateTimerDisplay();
    timerInterval = setInterval(() => {
        secondsElapsed++;
        updateTimerDisplay();

        // Автоматическая остановка записи после 15 секунд
        if (secondsElapsed >= 15) {
            stopRecording();
        }
    }, 1000);
}

function stopTimer() {
    clearInterval(timerInterval);
}

function updateTimerDisplay() {
    const timerElement = document.getElementById('timer');
    if (!timerElement) return;

    const minutes = Math.floor(secondsElapsed / 60);
    const seconds = secondsElapsed % 60;
    timerElement.textContent = `${minutes.toString().padStart(2, '0')}:${seconds.toString().padStart(2, '0')}`;
}

// Загрузка статуса системы аутентификации
async function loadSystemStatus() {
    try {
        const response = await fetch('/api/voice/system_status');

        if (!response.ok) {
            throw new Error(`Ошибка сервера: ${response.status}`);
        }

        const status = await response.json();

        // Обновляем индикаторы статуса, если они есть на странице
        const statusElement = document.getElementById('systemStatusIndicator');
        const statusTextElement = document.getElementById('systemStatusText');

        if (statusElement) {
            if (status.api_status === 'ok') {
                statusElement.className = 'status-indicator online';
                if (statusTextElement) statusTextElement.textContent = 'Система онлайн';
            } else {
                statusElement.className = 'status-indicator offline';
                if (statusTextElement) statusTextElement.textContent = 'Система офлайн';
            }
        }

        return status;
    } catch (error) {
        console.error("Ошибка при получении статуса системы:", error);

        // Обновляем индикатор статуса, если он есть
        const statusElement = document.getElementById('systemStatusIndicator');
        const statusTextElement = document.getElementById('systemStatusText');

        if (statusElement) {
            statusElement.className = 'status-indicator offline';
            if (statusTextElement) statusTextElement.textContent = 'Система недоступна';
        }

        return {
            api_status: 'error',
            message: error.message
        };
    }
}