document.addEventListener('DOMContentLoaded', function() {
    const languageSelect = document.getElementById('language');
    const levelSelect = document.getElementById('level');
    const difficultySelect = document.getElementById('difficulty');
    const textDisplay = document.getElementById('text-display');
    const loadTextBtn = document.getElementById('load-text-btn');
    const ttsBtn = document.getElementById('tts-btn');
    const recordBtn = document.getElementById('record-btn');
    const audioPlayback = document.getElementById('audio-playback');
    const evaluateBtn = document.getElementById('evaluate-btn');
    const resultsContainer = document.getElementById('results-container');
    const audioStatus = document.querySelector('.audio-status');

    let currentText = null;
    let mediaRecorder = null;
    let audioChunks = [];
    let isRecording = false;
    let audioBlob = null;
    let isSpeaking = false;

    loadTextBtn.addEventListener('click', function() {
        const language = languageSelect.value;
        const level = levelSelect.value;
        const difficulty = difficultySelect.value;

        textDisplay.textContent = 'Chargement du texte...';

        fetch(`/get_random_text?language=${language}&level=${level}&difficulty=${difficulty}`)
            .then(response => response.json())
            .then(data => {
                if (data.status === 'success') {
                    currentText = data.text;
                    textDisplay.textContent = currentText.content;

                    audioPlayback.style.display = 'none';
                    audioBlob = null;
                    evaluateBtn.disabled = true;
                    resultsContainer.style.display = 'none';

                    audioStatus.textContent = 'Cliquez sur le bouton pour commencer l\'enregistrement';
                    ttsBtn.disabled = false;
                } else {
                    textDisplay.textContent = 'Erreur: ' + data.message;
                }
            })
            .catch(error => {
                console.error('Error fetching text:', error);
                textDisplay.textContent = 'Erreur lors du chargement du texte. Veuillez réessayer.';
            });
    });

    ttsBtn.addEventListener('click', function() {
        if (!currentText) {
            alert('Veuillez d\'abord charger un texte.');
            return;
        }

        if (isSpeaking) {
            // Stop playback (handled by server)
            isSpeaking = false;
            ttsBtn.innerHTML = '<i class="fas fa-volume-up"></i> Écouter le texte';
            ttsBtn.classList.remove('playing');
            audioStatus.textContent = 'Cliquez sur le bouton pour commencer l\'enregistrement';
        } else {
            isSpeaking = true;
            ttsBtn.innerHTML = '<i class="fas fa-pause"></i> Pause';
            ttsBtn.classList.add('playing');
            audioStatus.textContent = 'Lecture du texte en cours...';

            fetch('/text-to-speech', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ text: currentText.content, language: currentText.language })
            })
            .then(response => response.json())
            .then(data => {
                if (data.status === 'success') {
                    const audio = new Audio(data.audio_url);
                    audio.onended = () => {
                        isSpeaking = false;
                        ttsBtn.innerHTML = '<i class="fas fa-volume-up"></i> Écouter le texte';
                        ttsBtn.classList.remove('playing');
                        audioStatus.textContent = 'Cliquez sur le bouton pour commencer l\'enregistrement';
                    };
                    audio.play();
                } else {
                    alert('Erreur TTS: ' + data.message);
                    isSpeaking = false;
                    ttsBtn.innerHTML = '<i class="fas fa-volume-up"></i> Écouter le texte';
                    ttsBtn.classList.remove('playing');
                }
            })
            .catch(error => {
                console.error('TTS error:', error);
                alert('Erreur lors de la synthèse vocale.');
                isSpeaking = false;
                ttsBtn.innerHTML = '<i class="fas fa-volume-up"></i> Écouter le texte';
                ttsBtn.classList.remove('playing');
            });
        }
    });

    recordBtn.addEventListener('click', function() {
        if (!currentText) {
            alert('Veuillez d\'abord charger un texte.');
            return;
        }

        if (isSpeaking) {
            isSpeaking = false;
            ttsBtn.innerHTML = '<i class="fas fa-volume-up"></i> Écouter le texte';
            ttsBtn.classList.remove('playing');
        }

        if (isRecording) {
            mediaRecorder.stop();
            mediaRecorder.stream.getTracks().forEach(track => track.stop());
        } else {
            navigator.mediaDevices.getUserMedia({ audio: true })
                .then(stream => {
                    audioChunks = [];
                    mediaRecorder = new MediaRecorder(stream, { mimeType: 'audio/webm' });

                    mediaRecorder.addEventListener('dataavailable', event => {
                        audioChunks.push(event.data);
                    });

                    mediaRecorder.addEventListener('stop', () => {
                        audioBlob = new Blob(audioChunks, { type: 'audio/webm' });
                        audioPlayback.src = URL.createObjectURL(audioBlob);
                        audioPlayback.style.display = 'block';

                        evaluateBtn.disabled = false;
                        isRecording = false;
                        recordBtn.classList.remove('recording');
                        recordBtn.innerHTML = '<i class="fas fa-microphone"></i>';
                        audioStatus.textContent = 'Enregistrement terminé. Vous pouvez l\'écouter ci-dessous.';
                    });

                    mediaRecorder.start();
                    isRecording = true;
                    recordBtn.classList.add('recording');
                    recordBtn.innerHTML = '<i class="fas fa-stop"></i>';
                    audioStatus.textContent = 'Enregistrement en cours...';
                })
                .catch(error => {
                    console.error('Microphone error:', error);
                    alert('Erreur lors de l\'accès au microphone.');
                });
        }
    });

    evaluateBtn.addEventListener('click', function() {
        if (!audioBlob || !currentText) {
            alert('Veuillez enregistrer votre lecture avant d\'évaluer.');
            return;
        }

        evaluateBtn.disabled = true;
        evaluateBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Évaluation en cours...';

        const formData = new FormData();
        formData.append('file', audioBlob, 'recording.webm');
        formData.append('reference_text', currentText.content);
        formData.append('text_id', currentText.text_id);
        formData.append('language', currentText.language);
        formData.append('level', currentText.level);
        formData.append('difficulty', currentText.difficulty);

        fetch('/evaluate', {
            method: 'POST',
            body: formData
        })
        .then(response => response.json())
        .then(data => {
            if (data.status === 'success') {
                document.getElementById('pronunciation-score').textContent = data.evaluation.pronunciation_accuracy + '/100';
                document.getElementById('reading-speed').textContent = data.words_per_minute + ' mots/min';
                document.getElementById('comprehension-score').textContent = data.evaluation.comprehension + '/100';
                document.getElementById('overall-score').textContent = data.evaluation.score + '/100';

                const feedbackContainer = document.getElementById('feedback-container');
                let feedback = '';
                if (data.llm_details.pronunciation.feedback) {
                    feedback += data.llm_details.pronunciation.feedback + '<br>';
                }
                if (data.llm_details.comprehension.feedback) {
                    feedback += data.llm_details.comprehension.feedback;
                }
                feedbackContainer.innerHTML = `
                    <h4>Commentaires:</h4>
                    <p>${feedback || 'Aucun commentaire disponible'}</p>
                `;

                resultsContainer.style.display = 'block';
            } else {
                alert('Erreur lors de l\'évaluation: ' + data.message);
            }
        })
        .catch(error => {
            console.error('Evaluation error:', error);
            alert('Erreur lors de l\'évaluation.');
        })
        .finally(() => {
            evaluateBtn.disabled = false;
            evaluateBtn.innerHTML = '<i class="fas fa-check-circle"></i> Évaluer';
        });
    });

    languageSelect.addEventListener('change', function() {
        if (this.value === 'ar') {
            textDisplay.style.direction = 'rtl';
            textDisplay.style.textAlign = 'right';
        } else {
            textDisplay.style.direction = 'ltr';
            textDisplay.style.textAlign = 'left';
        }
        if (isSpeaking) {
            isSpeaking = false;
            ttsBtn.innerHTML = '<i class="fas fa-volume-up"></i> Écouter le texte';
            ttsBtn.classList.remove('playing');
        }
        ttsBtn.disabled = true;
    });
});