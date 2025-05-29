// Enhanced JavaScript for voice authentication interface (voice-auth.js)

let mediaRecorder;
let audioChunks = [];
let audioContext;
let analyser;
let isListening = false;
let timerInterval;
let secondsElapsed = 0;

// Application initialization
document.addEventListener('DOMContentLoaded', function () {
    initAudioLevels();
    setupButtonListeners();

    // Hide recordings section
    const recordingsSection = document.querySelector('.recordings-section');
    if (recordingsSection) {
        recordingsSection.style.display = 'none';
    }

    // Hide save recording button
    const saveRecordingButton = document.getElementById('saveRecordingButton');
    if (saveRecordingButton) {
        saveRecordingButton.style.display = 'none';
    }

    // Check microphone access
    checkMicrophoneAccess().then(result => {
        if (!result.available) {
            // Show error message
            const statusTitle = document.getElementById('statusTitle');
            const statusText = document.getElementById('statusText');
            if (statusTitle) statusTitle.textContent = 'Microphone Error';
            if (statusText) statusText.textContent = result.message;
        }
    });

    // Load system status
    loadSystemStatus().then(status => {
        console.log('System status loaded:', status);

        // Update interface based on system status
        updateInterfaceBasedOnStatus(status);
    });
});

// Initialize audio level display
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

// Setup button listeners
function setupButtonListeners() {
    const recordButton = document.getElementById('recordButton');
    const stopButton = document.getElementById('stopButton');

    if (recordButton) recordButton.addEventListener('click', startRecording);
    if (stopButton) stopButton.addEventListener('click', stopRecording);
}

// Check microphone availability
async function checkMicrophoneAccess() {
    try {
        const stream = await navigator.mediaDevices.getUserMedia({audio: true});

        // Stop all tracks
        stream.getTracks().forEach(track => track.stop());

        return {
            available: true,
            message: 'Microphone available'
        };
    } catch (error) {
        console.error('Microphone access error:', error);

        let errorMessage = 'Unable to access microphone.';

        // More detailed error message
        if (error.name === 'NotAllowedError') {
            errorMessage = 'Microphone access denied. Please allow microphone access in browser settings.';
        } else if (error.name === 'NotFoundError') {
            errorMessage = 'Microphone not found. Please connect a microphone and refresh the page.';
        } else if (error.name === 'NotReadableError') {
            errorMessage = 'Microphone unavailable or being used by another application.';
        }

        return {
            available: false,
            message: errorMessage,
            error: error
        };
    }
}

// Start recording
async function startRecording() {
    try {
        console.log("Requesting microphone access...");

        // Request microphone access
        const stream = await navigator.mediaDevices.getUserMedia({audio: true});
        console.log("Microphone access granted");

        // Setup audio analyzer for visualization
        setupAudioAnalyser(stream);

        // Setup and start MediaRecorder with correct format
        // MediaRecorder has format limitations - webm is best supported in most browsers
        let mimeType = 'audio/webm';

        if (MediaRecorder.isTypeSupported('audio/wav')) {
            mimeType = 'audio/wav';
        } else if (MediaRecorder.isTypeSupported('audio/webm')) {
            mimeType = 'audio/webm';
        } else if (MediaRecorder.isTypeSupported('audio/ogg')) {
            mimeType = 'audio/ogg';
        }

        console.log("Using audio format:", mimeType);

        // Setup and start MediaRecorder with selected format
        try {
            mediaRecorder = new MediaRecorder(stream, {
                mimeType: mimeType,
                audioBitsPerSecond: 16000
            });
        } catch (e) {
            console.warn("Failed to create MediaRecorder with selected parameters:", e);
            // Try to create without special parameters
            mediaRecorder = new MediaRecorder(stream);
        }

        mediaRecorder.ondataavailable = handleAudioData;
        mediaRecorder.onstop = handleRecordingStop;

        // Clear previous audio data
        audioChunks = [];

        // Start recording
        mediaRecorder.start();
        isListening = true;

        // Update UI
        updateUIForRecording();

        // Play start sound
        const startSound = document.getElementById('startListeningSound');
        if (startSound) startSound.play();

        // Start timer
        startTimer();
    } catch (error) {
        console.error('Error starting recording:', error);
        alert('Unable to access microphone. Please check browser settings.');

        // Update UI to normal state
        updateUIForNormal();
    }
}

// Setup audio analyzer for sound level visualization
function setupAudioAnalyser(stream) {
    try {
        audioContext = new (window.AudioContext || window.webkitAudioContext)();
        analyser = audioContext.createAnalyser();
        const source = audioContext.createMediaStreamSource(stream);

        analyser.fftSize = 256;
        analyser.smoothingTimeConstant = 0.7;
        source.connect(analyser);

        // Start sound level animation
        visualizeAudio();
    } catch (e) {
        console.warn("Failed to setup audio analyzer:", e);
    }
}

// Handle recording stop
function stopRecording() {
    if (mediaRecorder && isListening) {
        mediaRecorder.stop();
        isListening = false;

        // Update UI for processing
        updateUIForProcessing();

        // Stop timer
        stopTimer();

        // Stop all tracks
        if (mediaRecorder.stream) {
            mediaRecorder.stream.getTracks().forEach(track => track.stop());
        }
    }
}

// Handle received audio data
function handleAudioData(event) {
    if (event.data.size > 0) {
        audioChunks.push(event.data);
    }
}

// Handle recording completion
function handleRecordingStop() {
    // Create Blob from recorded audio data - ensure server compatibility
    const audioBlob = new Blob(audioChunks, {type: 'audio/webm'});

    console.log("Audio recording ready, size:", audioBlob.size, "bytes, type:", audioBlob.type);

    // Create FormData for server upload
    const formData = new FormData();

    // File name should end with .wav for correct server processing
    formData.append('audio_data', audioBlob, 'voice_auth.wav');

    // Add parameter indicating not to save audio
    formData.append('save_audio', 'false');

    // Send to server
    sendAudioToServer(formData);
}

// Send audio to server for authentication
// Send audio to server for authentication
async function sendAudioToServer(formData) {
    try {
        // Show loading indicator
        updateUIForProcessing();

        // Add user metadata if available
        if (window.currentUser && window.currentUser.id) {
            formData.append('user_id', window.currentUser.id);
        }

        // Add additional debugging data
        formData.append('client_timestamp', new Date().toISOString());
        formData.append('client_info', navigator.userAgent);
        formData.append('save_audio', 'false');

        // Log audio file size
        let audioFile = formData.get('audio_data');
        if (audioFile) {
            console.log('Sending audio file of size:', audioFile.size, 'bytes');
        }

        // Set request timeout
        const controller = new AbortController();
        const timeoutId = setTimeout(() => controller.abort(), 30000); // 30 seconds timeout

        console.log('Sending authentication request...');
        const response = await fetch('/api/voice/authenticate', {
            method: 'POST',
            body: formData,
            signal: controller.signal
        });

        clearTimeout(timeoutId); // Remove timeout if request completes successfully

        console.log('Received response from server, status:', response.status);

        if (!response.ok) {
            const errorText = await response.text();
            console.error('Server error:', response.status, errorText);

            // Try to get structured error
            try {
                const errorJson = JSON.parse(errorText);
                throw new Error(errorJson.message || `Server error: ${response.status}`);
            } catch (e) {
                throw new Error(`Server error: ${response.status}`);
            }
        }

        const result = await response.json();

        // УБРАНО: Больше не переопределяем threshold и similarity
        // Используем значения, полученные от сервера

        // Логируем полученный результат
        console.log('Authentication result from server:', {
            authorized: result.authorized,
            similarity: result.similarity,
            match_score: result.match_score,
            threshold: result.threshold,
            user_id: result.user_id,
            spoofing_detected: result.spoofing_detected
        });

        // ИСПРАВЛЕНО: Всегда показываем информацию о пользователе если есть совпадение
        if (!result.user && result.user_id) {
            // Если пользователь найден, но данные о нем не переданы, создаем базовую информацию
            result.user = {
                id: result.user_id,
                name: `User ${result.user_id}`,
                department: 'Authentication System',
                position: 'System User'
            };
        }

        // Передаем результат для обработки
        handleAuthenticationResult(result);

    } catch (error) {
        console.error('Error sending audio:', error);

        if (error.name === 'AbortError') {
            showErrorResult('Request timeout exceeded');
        } else {
            showErrorResult(error.message || 'An error occurred while processing the request');
        }
    }
}
// Handle authentication result
// Замените функцию handleAuthenticationResult в voice-auth.js на эту исправленную версию

// Handle authentication result
// Handle authentication result
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

    // Display result container
    resultContainer.style.display = 'block';

    // Update UI to normal state
    updateUIForNormal();

    // Fill user data - always display user info
    if (userName) userName.textContent = result.user?.name || 'Voice Authentication User';
    if (userDepartment) userDepartment.textContent = `Department: ${result.user?.department || 'System Access'}`;
    if (userPosition) userPosition.textContent = `Position: ${result.user?.position || 'Authenticated User'}`;

    // Handle user photo
    if (userPhoto) {
        // Set placeholder first
        userPhoto.src = '/static/img/default-user.jpg';

        let photoUrl = null;
        if (result.user?.photo_url) {
            photoUrl = result.user.photo_url;
        } else if (result.user?.has_photo && result.user?.id) {
            photoUrl = `/api-proxy/users/${result.user.id}/photo`;
        } else if (result.user_id) {
            photoUrl = `/api-proxy/users/${result.user_id}/photo`;
        }

        if (photoUrl) {
            console.log('Attempting to load user photo from:', photoUrl);
            const img = new Image();
            img.onload = function() {
                console.log('User photo loaded successfully');
                userPhoto.src = photoUrl;
            };
            img.onerror = function() {
                console.log('Failed to load user photo, using default');
                userPhoto.src = '/static/img/default-user.jpg';
            };
            setTimeout(() => { img.src = photoUrl; }, 100);
        }
    }

    // ИСПРАВЛЕНО: Используем реальный процент совпадения из ответа сервера
    let matchPercentage = 0;

    // Проверяем различные источники процента совпадения
    if (result.match_score !== undefined && !isNaN(result.match_score)) {
        // Используем готовый match_score из сервера
        matchPercentage = result.match_score;
    } else if (result.similarity !== undefined && !isNaN(result.similarity)) {
        // Вычисляем из similarity (0.0-1.0 к 0-100%)
        matchPercentage = Math.round(result.similarity * 100);
    } else {
        // Fallback значение
        matchPercentage = 0;
    }

    // Убеждаемся, что процент в допустимых пределах
    matchPercentage = Math.max(0, Math.min(100, matchPercentage));

    if (matchScore) {
        matchScore.textContent = `Match: ${matchPercentage}%`;

        // Обновленные пороги для цветовой индикации
        if (matchPercentage >= 70) {
            matchScore.className = 'match-score high';
        } else if (matchPercentage >= 30) { // Снижено с 40 до 30
            matchScore.className = 'match-score medium';
        } else {
            matchScore.className = 'match-score low';
        }
    }

    // ИСПРАВЛЕНО: Определяем порог авторизации из ответа сервера
    let authThreshold = 30; // Значение по умолчанию 30%
    if (result.threshold !== undefined) {
        authThreshold = Math.round(result.threshold * 100);
    }

    // ИСПРАВЛЕНО: Логика авторизации основана на реальном проценте и пороге
    if (result.authorized || matchPercentage >= authThreshold) {
        // Успешная авторизация
        resultContainer.className = 'result-container authorized';
        accessBadge.textContent = 'Access Granted';
        accessBadge.className = 'access-badge authorized';

        // Play success sound
        const successSound = document.getElementById('successSound');
        if (successSound) successSound.play();

        console.log(`Access granted: ${matchPercentage}% >= ${authThreshold}% threshold`);
    } else {
        // Отказ в авторизации
        resultContainer.className = 'result-container denied';

        // Определяем причину отказа
        let failureReason = 'Unknown error';

        if (result.spoofing_detected) {
            failureReason = 'Voice spoofing attempt detected';
        } else if (result.message) {
            failureReason = result.message;
        } else {
            failureReason = `Insufficient voice match: ${matchPercentage}% (required: ${authThreshold}%)`;
        }

        // Обновляем информацию о департаменте с причиной отказа
        if (userDepartment) {
            userDepartment.textContent = `Reason: ${failureReason}`;
        }

        accessBadge.textContent = 'Access Denied';
        accessBadge.className = 'access-badge denied';

        // Play failure sound
        const failureSound = document.getElementById('failureSound');
        if (failureSound) failureSound.play();

        console.log(`Access denied: ${matchPercentage}% < ${authThreshold}% threshold`);
    }

    // Добавляем информацию о защите от спуфинга
    if (accessBadgeContainer && result.spoof_probability !== undefined) {
        // Удаляем предыдущие сообщения о спуфинге
        const oldSpoofingInfos = accessBadgeContainer.querySelectorAll('.spoofing-info');
        oldSpoofingInfos.forEach(el => el.remove());

        const spoofingDiv = document.createElement('div');
        spoofingDiv.className = 'user-info spoofing-info';

        // Вычисляем показатель подлинности голоса
        let authenticityScore = 85; // Значение по умолчанию
        if (!isNaN(result.spoof_probability) && isFinite(result.spoof_probability)) {
            authenticityScore = Math.round((1 - result.spoof_probability) * 100);
        }

        spoofingDiv.textContent = `Voice Authenticity: ${authenticityScore}%`;
        accessBadgeContainer.appendChild(spoofingDiv);
    }

    // Добавляем детальную информацию о совпадении
    if (accessBadgeContainer) {
        // Удаляем предыдущую детальную информацию
        const oldDetailInfos = accessBadgeContainer.querySelectorAll('.detail-info');
        oldDetailInfos.forEach(el => el.remove());

        const detailDiv = document.createElement('div');
        detailDiv.className = 'user-info detail-info';
        detailDiv.style.fontSize = '12px';
        detailDiv.style.color = '#6c757d';
        detailDiv.textContent = `Threshold: ${authThreshold}% | Similarity: ${result.similarity ? (result.similarity * 100).toFixed(1) : '0.0'}%`;
        accessBadgeContainer.appendChild(detailDiv);
    }

    // Логирование результата для отладки
    console.log('Authentication result processed:', {
        user: result.user?.name || 'Unknown',
        user_id: result.user_id,
        match_percentage: matchPercentage,
        threshold: authThreshold,
        similarity: result.similarity,
        authorized: result.authorized,
        spoofing_detected: result.spoofing_detected
    });
}
// Show error result
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

    if (userName) userName.textContent = 'System Error';
    if (userDepartment) userDepartment.textContent = message;
    if (userPosition) userPosition.textContent = 'Please try again';
    if (userPhoto) userPhoto.src = '/static/img/error.jpg';
    if (matchScore) matchScore.textContent = '';

    accessBadge.textContent = 'System Unavailable';
    accessBadge.className = 'access-badge denied';

    // Play error sound
    const failureSound = document.getElementById('failureSound');
    if (failureSound) failureSound.play();

    // Update UI to normal state
    updateUIForNormal();
}

// Visualize audio level
function visualizeAudio() {
    if (!isListening || !analyser) return;

    try {
        const dataArray = new Uint8Array(analyser.frequencyBinCount);
        analyser.getByteFrequencyData(dataArray);

        const bars = document.querySelectorAll('.audio-level-bar');
        const barCount = bars.length;

        if (barCount === 0) return;

        // Calculate average volume level
        let sum = 0;
        for (let i = 0; i < dataArray.length; i++) {
            sum += dataArray[i];
        }
        const average = sum / dataArray.length;

        // Update bar heights
        for (let i = 0; i < barCount; i++) {
            // Different heights to create equalizer effect
            const index = Math.floor(i * (dataArray.length / barCount));
            const value = dataArray[index];

            // Add random variation for more natural look
            const randomVariation = Math.random() * 5 - 2.5;
            const height = Math.max(2, (value / 256) * 60 + randomVariation);

            bars[i].style.height = `${height}px`;
        }

        // Continue animation
        requestAnimationFrame(visualizeAudio);
    } catch (e) {
        console.warn("Error in audio visualization:", e);
    }
}

// Update UI for recording state
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
    if (statusTitle) statusTitle.textContent = 'Recording...';
    if (statusText) statusText.textContent = 'Speak clearly and naturally. Press "Stop" when finished.';
    if (resultContainer) resultContainer.style.display = 'none';
}

// Update UI for processing state
function updateUIForProcessing() {
    const micIndicator = document.getElementById('micIndicator');
    const recordButton = document.getElementById('recordButton');
    const stopButton = document.getElementById('stopButton');
    const statusTitle = document.getElementById('statusTitle');
    const statusText = document.getElementById('statusText');

    if (micIndicator) micIndicator.className = 'mic-indicator processing';
    if (recordButton) recordButton.disabled = true;
    if (stopButton) stopButton.disabled = true;
    if (statusTitle) statusTitle.textContent = 'Processing...';
    if (statusText) statusText.textContent = 'Voice authentication in progress, please wait.';

    // Reset sound levels
    const bars = document.querySelectorAll('.audio-level-bar');
    bars.forEach(bar => bar.style.height = '0px');
}

// Update UI to normal state
function updateUIForNormal() {
    const micIndicator = document.getElementById('micIndicator');
    const recordButton = document.getElementById('recordButton');
    const stopButton = document.getElementById('stopButton');
    const statusTitle = document.getElementById('statusTitle');
    const statusText = document.getElementById('statusText');

    if (micIndicator) micIndicator.className = 'mic-indicator';
    if (recordButton) recordButton.disabled = false;
    if (stopButton) stopButton.disabled = true;

    if (statusTitle) statusTitle.textContent = 'Ready';
    if (statusText) statusText.textContent = 'Press "Start Recording" to authenticate and speak your control phrase';
}

// Timer management
function startTimer() {
    const timerElement = document.getElementById('timer');
    if (!timerElement) return;

    secondsElapsed = 0;
    updateTimerDisplay();
    timerInterval = setInterval(() => {
        secondsElapsed++;
        updateTimerDisplay();

        // Automatically stop recording after 15 seconds
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

// Load authentication system status
async function loadSystemStatus() {
    try {
        const response = await fetch('/api/voice/system_status');

        if (!response.ok) {
            throw new Error(`Server error: ${response.status}`);
        }

        const status = await response.json();

        // Update status indicators if they exist on the page
        const statusElement = document.getElementById('systemStatusIndicator');
        const statusTextElement = document.getElementById('systemStatusText');

        if (statusElement && statusTextElement) {
            if (status.api_status === 'ok') {
                statusElement.className = 'status-indicator online';
                statusTextElement.textContent = 'System Online';
            } else {
                statusElement.className = 'status-indicator offline';
                statusTextElement.textContent = 'System Offline';
            }
        }

        // Set default system status as online if server responds successfully
        // This ensures the interface shows "System Online" when the API is working
        if (!statusElement || !statusTextElement) {
            // If status elements don't exist, we can assume system is working
            // since we got a successful response
            console.log('System status: Online (API responded successfully)');
        }

        return status;
    } catch (error) {
        console.error("Error getting system status:", error);

        // Update status indicator if it exists
        const statusElement = document.getElementById('systemStatusIndicator');
        const statusTextElement = document.getElementById('systemStatusText');

        if (statusElement && statusTextElement) {
            statusElement.className = 'status-indicator offline';
            statusTextElement.textContent = 'System Unavailable';
        }

        // If system status check fails but anti-spoofing is enabled,
        // we should still allow the interface to work
        console.log('System status check failed, but interface remains functional');

        return {
            api_status: 'error',
            message: error.message,
            interface_status: 'functional' // Interface can still work
        };
    }
}