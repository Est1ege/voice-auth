document.addEventListener('DOMContentLoaded', function() {
    // Добавляем стилизацию для страницы аутентификации
    const styleElement = document.createElement('style');
    styleElement.textContent = `
        .recordings-section {
            display: none; /* Скрываем раздел управления аудиозаписями */
        }
        #saveRecordingButton {
            display: none; /* Скрываем кнопку сохранения */
        }
        .auth-info {
            margin-top: 20px;
            padding: 15px;
            background-color: #f8f9fa;
            border-radius: 8px;
            text-align: center;
        }
        .auth-info p {
            margin-bottom: 5px;
            font-size: 14px;
        }
    `;
    document.head.appendChild(styleElement);

    // Добавляем информационный блок о настройках системы
    const authContainer = document.querySelector('.auth-container');
    if (authContainer) {
        const authInfoDiv = document.createElement('div');
        authInfoDiv.className = 'auth-info';
        authInfoDiv.innerHTML = `
            <p><strong>Информация о системе:</strong></p>
            <p>Текущий порог распознавания: 0.7 (70%)</p>
            <p>Режим сохранения аудио: отключен</p>
        `;

        // Вставляем перед управляющими элементами
        const controlPanel = document.querySelector('.control-panel');
        if (controlPanel) {
            authContainer.insertBefore(authInfoDiv, controlPanel);
        } else {
            authContainer.appendChild(authInfoDiv);
        }
    }
});