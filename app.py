from flask import Flask, request, jsonify, render_template
import os
import json
import librosa
import numpy as np
from jiwer import wer, cer
from fastdtw import fastdtw
from transformers import pipeline, AutoProcessor, AutoModelForSpeechSeq2Seq
import torch
from gtts import gTTS
import uuid
import tempfile
import logging
from dotenv import load_dotenv
import threading
import time
import soundfile as sf
from pydub import AudioSegment

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()
hf_token = os.getenv('HF_TOKEN')

app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
TEXT_DB_PATH = 'texts/texts.json'
AUDIO_OUTPUT_FOLDER = 'static/audio'
REFERENCE_AUDIO_FOLDER = 'reference_audio'
os.makedirs(AUDIO_OUTPUT_FOLDER, exist_ok=True)
os.makedirs(REFERENCE_AUDIO_FOLDER, exist_ok=True)
MAX_AUDIO_DURATION = 15  # seconds

# Load text database
try:
    with open(TEXT_DB_PATH, 'r', encoding='utf-8') as f:
        text_db = json.load(f)
    logger.info("Text database loaded successfully")
except Exception as e:
    logger.error(f"Failed to load text database: {e}")
    raise

# Generate reference audio for a single text
def generate_reference_audio(text, language, text_id):
    ref_audio_path = os.path.join(REFERENCE_AUDIO_FOLDER, f"ref_{text_id}.wav")
    if not os.path.exists(ref_audio_path):
        try:
            tts = gTTS(text=text, lang=language)
            temp_mp3 = os.path.join(tempfile.gettempdir(), f"temp_{uuid.uuid4()}.mp3")
            tts.save(temp_mp3)
            audio = AudioSegment.from_mp3(temp_mp3).set_frame_rate(16000).set_channels(1)
            audio.export(ref_audio_path, format="wav")
            os.remove(temp_mp3)
            logger.info(f"Generated reference audio for text_id {text_id}")
        except Exception as e:
            logger.error(f"Error generating reference audio for text_id {text_id}: {e}")
            raise
    return ref_audio_path

# Load Whisper model
try:
    device = torch.device("cpu")
    logger.info("Loading Whisper model on CPU")
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        "bofenghuang/whisper-small-cv11-french",
        use_safetensors=False,
        token=hf_token if hf_token else None
    ).to(device)
    processor = AutoProcessor.from_pretrained(
        "bofenghuang/whisper-small-cv11-french",
        language="french",
        task="transcribe",
        token=hf_token if hf_token else None
    )
    asr_pipeline = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        device=device,
        generate_kwargs={"num_beams": 1, "max_new_tokens": 50}
    )
    asr_pipeline.model.config.forced_decoder_ids = asr_pipeline.tokenizer.get_decoder_prompt_ids(language="fr", task="transcribe")
    logger.info("Whisper model loaded successfully")
except Exception as e:
    logger.error(f"Failed to load bofenghuang/whisper-small-cv11-french: {e}")
    raise

# Audio analysis functions
def convert_to_wav(input_path, output_path):
    try:
        audio = AudioSegment.from_file(input_path)
        audio = audio.set_frame_rate(16000).set_channels(1)
        audio.export(output_path, format="wav")
        logger.info(f"Converted {input_path} to WAV")
    except Exception as e:
        logger.error(f"Error converting audio to WAV: {e}")
        raise

def get_audio_duration(audio_path):
    try:
        with sf.SoundFile(audio_path) as f:
            duration = len(f) / f.samplerate
        logger.info(f"Duration via soundfile: {duration:.2f} seconds")
        return duration
    except Exception as e:
        logger.error(f"Error getting duration with soundfile: {e}")
        try:
            y, sr = librosa.load(audio_path, sr=16000)
            duration = len(y) / sr
            logger.info(f"Duration via librosa: {duration:.2f} seconds")
            return duration
        except Exception as e2:
            logger.error(f"Error getting duration with librosa: {e2}")
            raise

def extract_audio_features(audio_path):
    try:
        start_time = time.time()
        y, sr = librosa.load(audio_path, sr=16000)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        logger.info(f"Feature extraction took {time.time() - start_time:.2f} seconds")
        return {'mfccs': mfccs, 'y': y, 'sr': sr, 'tempo': tempo}
    except Exception as e:
        logger.error(f"Error extracting audio features: {e}")
        raise

def compare_pronunciation(reference_mfccs, user_mfccs, ref_duration, ref_words):
    try:
        start_time = time.time()
        min_len = min(reference_mfccs.shape[1], user_mfccs.shape[1])
        ref_mfccs = reference_mfccs[:, :min_len]
        usr_mfccs = user_mfccs[:, :min_len]
        dist, path = fastdtw(ref_mfccs.T, usr_mfccs.T, dist=lambda x, y: np.linalg.norm(x - y))
        ref_norm = np.linalg.norm(ref_mfccs)
        norm_dist = dist / ref_norm if ref_norm > 0 else dist
        score = max(0, 100 - norm_dist * 2)
        logger.info(f"DTW distance: {dist:.2f}, Normalized: {norm_dist:.2f}, Score: {score:.2f}")
        errors = []
        path = np.array(path)
        for i in range(1, len(path)):
            if abs(path[i][0] - path[i-1][0]) > 1 or abs(path[i][1] - path[i-1][1]) > 1:
                time_sec = (path[i][0] / ref_mfccs.shape[1]) * ref_duration
                word_idx = int(len(ref_words) * (time_sec / ref_duration))
                if word_idx < len(ref_words):
                    errors.append({'time': round(time_sec, 2), 'word': ref_words[word_idx]})
        logger.info(f"Pronunciation comparison took {time.time() - start_time:.2f} seconds")
        return round(score, 2), errors
    except Exception as e:
        logger.error(f"Error comparing pronunciation: {e}")
        raise

def evaluate_text_accuracy(reference_text, transcribed_text, result):
    try:
        start_time = time.time()
        ref = reference_text.lower().strip()
        trans = transcribed_text.lower().strip()
        wer_score = wer(ref, trans)
        cer_score = cer(ref, trans)
        text_accuracy = max(0, 100 * (1 - wer_score))
        result['wer_accuracy'] = round(text_accuracy, 2)
        result['cer_accuracy'] = round(100 * (1 - cer_score), 2)
        logger.info(f"Text accuracy (WER) took {time.time() - start_time:.2f} seconds")
    except Exception as e:
        logger.error(f"Error evaluating text accuracy: {e}")
        raise

def get_speed_score(words_per_minute, tempo):
    if words_per_minute > 180 or tempo > 120:
        return 60  # Too fast
    elif words_per_minute < 80 or tempo < 80:
        return 60  # Too slow
    else:
        return 100  # Appropriate

def generate_feedback(pronunciation_score, text_accuracy, audio_features, words_per_minute, pronunciation_errors):
    try:
        feedback = {
            'pronunciation': '',
            'reading_speed': '',
            'comprehension': '',
            'error_details': []
        }
        if pronunciation_score >= 80:
            feedback['pronunciation'] = "Excellente prononciation ! Continuez ainsi."
        elif pronunciation_score >= 60:
            feedback['pronunciation'] = "Bonne prononciation, mais certains mots peuvent être améliorés."
        else:
            feedback['pronunciation'] = "La prononciation nécessite plus de pratique."
        if pronunciation_errors:
            feedback['error_details'] = [
                f"Mot '{e['word']}' mal prononcé vers {e['time']}s" for e in pronunciation_errors[:3]
            ]
        if words_per_minute > 180 or audio_features['tempo'] > 120:
            feedback['reading_speed'] = "Lecture rapide, ralentissez pour plus de clarté."
        elif words_per_minute < 80 or audio_features['tempo'] < 80:
            feedback['reading_speed'] = "Lecture lente, accélérez légèrement."
        else:
            feedback['reading_speed'] = "Vitesse appropriée, bien joué !"
        if text_accuracy >= 80:
            feedback['comprehension'] = "Très bonne compréhension."
        elif text_accuracy >= 60:
            feedback['comprehension'] = "Bonne compréhension, quelques erreurs."
        else:
            feedback['comprehension'] = "Relisez attentivement."
        return feedback
    except Exception as e:
        logger.error(f"Error generating feedback: {e}")
        raise

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/get_random_text')
def get_random_text():
    try:
        language = request.args.get('language', 'fr')
        level = request.args.get('level', '6')
        difficulty = request.args.get('difficulty', 'medium')
        matching_texts = [
            text for text in text_db
            if text['language'] == language and text['level'] == level and text['difficulty'] == difficulty
        ]
        if not matching_texts:
            return jsonify({'status': 'error', 'message': 'Aucun texte disponible.'})
        selected_text = np.random.choice(matching_texts)
        generate_reference_audio(selected_text['content'], selected_text['language'], selected_text['id'])
        return jsonify({
            'status': 'success',
            'text': {
                'text_id': selected_text['id'],
                'content': selected_text['content'],
                'language': selected_text['language'],
                'level': selected_text['level'],
                'difficulty': selected_text['difficulty']
            }
        })
    except Exception as e:
        logger.error(f"Error getting random text: {e}")
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/text-to-speech', methods=['POST'])
def text_to_speech():
    try:
        data = request.get_json()
        text = data.get('text')
        language = data.get('language', 'fr')
        if not text:
            return jsonify({'status': 'error', 'message': 'Texte requis.'})
        tts = gTTS(text=text, lang=language)
        audio_filename = f"tts_{uuid.uuid4()}.mp3"
        audio_path = os.path.join(AUDIO_OUTPUT_FOLDER, audio_filename)
        tts.save(audio_path)
        audio_url = f"/static/audio/{audio_filename}"
        return jsonify({'status': 'success', 'audio_url': audio_url})
    except Exception as e:
        logger.error(f"Error in text-to-speech: {e}")
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/evaluate', methods=['POST'])
def evaluate():
    start_time = time.time()
    temp_audio_path = None
    wav_audio_path = None
    try:
        if 'file' not in request.files or not request.form.get('reference_text'):
            return jsonify({'status': 'error', 'message': 'Fichier audio ou texte manquant.'})
        audio_file = request.files['file']
        reference_text = request.form.get('reference_text')
        text_id = request.form.get('text_id')
        language = request.form.get('language')
        level = request.form.get('level')
        difficulty = request.form.get('difficulty')

        # Save audio
        temp_audio_path = os.path.join(UPLOAD_FOLDER, f"temp_{uuid.uuid4()}.{audio_file.filename.split('.')[-1]}")
        audio_file.save(temp_audio_path)

        # Convert to WAV
        wav_audio_path = os.path.join(UPLOAD_FOLDER, f"temp_{uuid.uuid4()}.wav")
        convert_to_wav(temp_audio_path, wav_audio_path)

        # Check audio duration
        duration = get_audio_duration(wav_audio_path)
        logger.info(f"Audio duration: {duration:.2f} seconds")
        if duration > MAX_AUDIO_DURATION:
            return jsonify({'status': 'error', 'message': f"Audio trop long ({duration:.1f}s). Limite: {MAX_AUDIO_DURATION}s."})
        if duration < 0.1:
            return jsonify({'status': 'error', 'message': 'Audio trop court ou invalide.'})

        # Transcribe
        transcribe_start = time.time()
        transcription = asr_pipeline(wav_audio_path)['text']
        logger.info(f"Transcription: {transcription}")
        logger.info(f"Transcription took {time.time() - transcribe_start:.2f} seconds")

        # Extract features
        audio_features = extract_audio_features(wav_audio_path)

        # Load reference audio
        ref_audio_path = os.path.join(REFERENCE_AUDIO_FOLDER, f"ref_{text_id}.wav")
        if not os.path.exists(ref_audio_path):
            logger.warning(f"Reference audio for text_id {text_id} not found, using fallback score")
            pronunciation_score = 80
            pronunciation_errors = []
        else:
            ref_features = extract_audio_features(ref_audio_path)
            ref_duration = get_audio_duration(ref_audio_path)
            ref_words = reference_text.lower().split()
            pronunciation_score, pronunciation_errors = compare_pronunciation(
                ref_features['mfccs'], audio_features['mfccs'], ref_duration, ref_words
            )

        # Text accuracy
        result = {}
        alignment_thread = threading.Thread(target=evaluate_text_accuracy, args=(reference_text, transcription, result))
        alignment_thread.start()
        alignment_thread.join()

        text_accuracy = result.get('wer_accuracy', 80)
        cer_accuracy = result.get('cer_accuracy', 80)
        word_count = len(reference_text.split())
        logger.info(f"Word count: {word_count}")
        words_per_minute = (word_count / duration) * 60 if duration > 0 else 0
        logger.info(f"Words per minute: {words_per_minute:.2f}")

        # Speed score
        speed_score = get_speed_score(words_per_minute, audio_features['tempo'])

        # Feedback
        feedback = generate_feedback(pronunciation_score, text_accuracy, audio_features, words_per_minute, pronunciation_errors)

        # Overall score
        overall_score = round(
            0.5 * pronunciation_score + 0.3 * text_accuracy + 0.2 * speed_score,
            2
        )

        evaluation = {
            'pronunciation_accuracy': pronunciation_score,
            'reading_speed': feedback['reading_speed'],
            'comprehension': text_accuracy,
            'cer_accuracy': cer_accuracy,
            'speed_score': speed_score,
            'score': overall_score
        }
        response = {
            'status': 'success',
            'evaluation': evaluation,
            'words_per_minute': round(words_per_minute, 2),
            'transcription': transcription,  # Add transcription to response
            'reference_text': reference_text,  # Add reference text for comparison
            'llm_details': {
                'pronunciation': {
                    'feedback': feedback['pronunciation'],
                    'errors': feedback['error_details']
                },
                'comprehension': {'feedback': feedback['comprehension']}
            }
        }
        logger.info(f"Total evaluation took {time.time() - start_time:.2f} seconds")
        return jsonify(response)
    except Exception as e:
        logger.error(f"Error in evaluation: {e}")
        return jsonify({'status': 'error', 'message': str(e)})
    finally:
        if temp_audio_path and os.path.exists(temp_audio_path):
            os.remove(temp_audio_path)
        if wav_audio_path and os.path.exists(wav_audio_path):
            os.remove(wav_audio_path)

@app.route('/list_texts')
def list_texts():
    try:
        return jsonify({'status': 'success', 'texts': text_db})
    except Exception as e:
        logger.error(f"Error listing texts: {e}")
        return jsonify({'status': 'error', 'message': str(e)})

if __name__ == '__main__':
    app.run(debug=True)