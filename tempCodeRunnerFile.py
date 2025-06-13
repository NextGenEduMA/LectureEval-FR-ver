from flask import Flask, request, jsonify, render_template, send_file
import os
import json
import librosa
import numpy as np
from jiwer import wer, cer, compute_measures
from fastdtw import fastdtw
from transformers import pipeline, AutoProcessor, AutoModelForSpeechSeq2Seq
import torch
from gtts import gTTS
import uuid
import tempfile
import logging
import soundfile as sf
from pydub import AudioSegment
import requests
from rag_retriever import RAGRetriever

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
from dotenv import load_dotenv
load_dotenv()
hf_token = os.getenv('HF_TOKEN')
gemini_api_key = os.getenv('GEMINI_API_KEY')
hf_api_key = os.getenv('HF_API_KEY')

app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = 'Uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
TEXT_DB_PATH = 'texts/texts.json'
AUDIO_OUTPUT_FOLDER = 'static/audio'
REFERENCE_AUDIO = 'reference_audio'
os.makedirs(AUDIO_OUTPUT_FOLDER, exist_ok=True)
os.makedirs(REFERENCE_AUDIO, exist_ok=True)
MAX_AUDIO_DURATION = 15  # seconds
MIN_SNR = 10  # dB
MIN_SAMPLE_RATE = 16000  # Hz

# Initialize RAG retriever
rag_retriever = RAGRetriever(corpus_path="rag_corpus", persist_directory="faiss_db")

# Load text database
try:
    with open(TEXT_DB_PATH, 'r', encoding='utf-8') as f:
        text_db = json.load(f)
    logger.info("Text database loaded successfully")
except Exception as e:
    logger.error(f"Failed to load text database: {e}")
    raise

# Initialize Whisper model
device = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"Using device: {device}")
processor = AutoProcessor.from_pretrained("openai/whisper-tiny", token=hf_token)
model = AutoModelForSpeechSeq2Seq.from_pretrained("openai/whisper-tiny", token=hf_token).to(device)

def generate_reference_audio(text, output_path):
    try:
        tts = gTTS(text=text, lang='fr')
        tts.save(output_path)
        logger.info(f"Generated reference audio at {output_path}")
    except Exception as e:
        logger.error(f"Error generating reference audio: {e}")
        raise

def check_audio_quality(audio_path):
    try:
        y, sr = librosa.load(audio_path, sr=None)
        if sr < MIN_SAMPLE_RATE:
            return {"is_valid": False, "messages": [f"Sample rate {sr}Hz is below minimum {MIN_SAMPLE_RATE}Hz"]}
        if len(y) / sr > MAX_AUDIO_DURATION:
            return {"is_valid": False, "messages": [f"Audio duration {len(y)/sr:.2f}s exceeds {MAX_AUDIO_DURATION}s"]}
        signal_power = np.mean(y ** 2)
        noise = np.random.normal(0, 0.01, len(y))
        noise_power = np.mean(noise ** 2)
        snr = 10 * np.log10(signal_power / noise_power) if noise_power > 0 else float('inf')
        if snr < MIN_SNR:
            return {"is_valid": False, "messages": [f"SNR {snr:.2f}dB is below minimum {MIN_SNR}dB"]}
        return {"is_valid": True, "messages": ["Audio quality is acceptable"]}
    except Exception as e:
        logger.error(f"Error checking audio quality: {e}")
        return {"is_valid": False, "messages": [f"Error processing audio: {str(e)}"]}

def analyze_pronunciation(reference_audio, user_audio):
    try:
        y_ref, sr_ref = librosa.load(reference_audio, sr=None)
        y_user, sr_user = librosa.load(user_audio, sr=None)
        if sr_ref != sr_user:
            y_user = librosa.resample(y_user, orig_sr=sr_user, target_sr=sr_ref)
        mfcc_ref = librosa.feature.mfcc(y=y_ref, sr=sr_ref, n_mfcc=13)
        mfcc_user = librosa.feature.mfcc(y=y_user, sr=sr_ref, n_mfcc=13)
        distance, _ = fastdtw(mfcc_ref.T, mfcc_user.T)
        max_distance = max(mfcc_ref.shape[1], mfcc_user.shape[1]) * 100
        pronunciation_score = max(0, 100 - (distance / max_distance * 100))
        errors = []
        if pronunciation_score < 80:
            errors.append({"word": "unknown", "time": 0, "confidence": 0.5})
        return pronunciation_score, errors
    except Exception as e:
        logger.error(f"Error analyzing pronunciation: {e}")
        return 50, [{"word": "unknown", "time": 0, "confidence": 0.5}]

def analyze_text_accuracy(reference_text, transcription):
    try:
        reference_text = reference_text.lower().strip()
        transcription = transcription.lower().strip()
        if not transcription:
            return 0, {"missed": reference_text.split(), "extra": [], "substituted": []}
        measures = compute_measures(reference_text, transcription)
        text_accuracy = max(0, 100 - measures['wer'] * 100)
        missed_words = []
        extra_words = []
        for op in measures.get('ops', []):
            if op['type'] == 'delete':
                missed_words.append(op['ref'])
            elif op['type'] == 'insert':
                extra_words.append(op['hyp'])
        return text_accuracy, {"missed": missed_words, "extra": extra_words, "substituted": []}
    except Exception as e:
        logger.error(f"Error analyzing text accuracy: {e}")
        return 50, {"missed": [], "extra": [], "substituted": []}

def calculate_reading_speed(audio_path, transcription):
    try:
        y, sr = librosa.load(audio_path, sr=None)
        duration = len(y) / sr
        word_count = len(transcription.split())
        words_per_minute = (word_count / duration) * 60 if duration > 0 else 0
        return words_per_minute
    except Exception as e:
        logger.error(f"Error calculating reading speed: {e}")
        return 0

def transcribe_audio(audio_path):
    try:
        audio, sr = sf.read(audio_path)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if sr != 16000:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
        inputs = processor(audio, sampling_rate=16000, return_tensors="pt").to(device)
        generated_ids = model.generate(inputs["input_features"])
        transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        logger.info(f"Transcribed audio: {transcription}")
        return transcription
    except Exception as e:
        logger.error(f"Error transcribing audio: {e}")
        return ""

def generate_fallback_feedback(pronunciation_score, text_accuracy, words_per_minute, pronunciation_errors, word_errors):
    feedback = f"**Gemini (Fallback)**: Ton score de prononciation est {pronunciation_score}/100, et ta vitesse est {words_per_minute:.1f} mots/min. Continue à pratiquer en lisant à voix haute !\n"
    feedback += f"**LLaMA (Fallback)**: Super effort ! Ta compréhension est à {text_accuracy}/100. Lis lentement pour améliorer la clarté."
    return {
        'feedback': feedback,
        'error_details': {
            'mispronounced': [f"Mot {e['word']}" for e in pronunciation_errors],
            'missed': word_errors.get('missed', []),
            'extra': word_errors.get('extra', [])
        }
    }

def generate_grok_feedback(pronunciation_score, text_accuracy, words_per_minute, pronunciation_errors, transcription, reference_text, quality_info, word_errors, level):
    try:
        missed_words = word_errors.get('missed', [])
        extra_words = word_errors.get('extra', [])
        mispronounced = [f"'{e['word']}' à {e['time']}s" for e in pronunciation_errors]
        quality_messages = quality_info.get('messages', ['Qualité audio non évaluée'])

        query = f"Conseils pour améliorer la prononciation, la compréhension, et la vitesse de lecture en français."
        retrieved_docs = rag_retriever.retrieve(query, level, n_results=2)
        retrieved_context = "\n".join([f"- {doc}" for doc in retrieved_docs]) if retrieved_docs else "Aucun conseil supplémentaire disponible."

        gemini_prompt = """
Vous êtes un professeur de français pour des élèves du primaire (niveau {level}). Donnez un retour CONCIS (<80 mots), POSITIF, et MOTIVANT, adapté aux enfants. Utilisez UNIQUEMENT les données et le contexte fournis.

Incluez :
- Points forts (prononciation, vitesse, compréhension).
- Conseils généraux pour s'améliorer, basés sur le contexte.
- Commentaire sur la qualité audio.

Données :
- Prononciation : {pronunciation_score}/100
- Compréhension (WER) : {text_accuracy}/100
- Vitesse : {words_per_minute} mots/min
- Mots mal prononcés : {mispronounced}
- Mots manquants : {missed_words}
- Mots supplémentaires : {extra_words}
- Qualité audio : {quality_messages}
- Transcription : "{transcription}"
- Référence : "{reference_text}"
Contexte :
{retrieved_context}
"""
        gemini_prompt = gemini_prompt.format(
            level=level,
            pronunciation_score=pronunciation_score,
            text_accuracy=text_accuracy,
            words_per_minute=words_per_minute,
            mispronounced=', '.join(mispronounced) or 'Aucun',
            missed_words=', '.join(missed_words) or 'Aucun',
            extra_words=', '.join(extra_words) or 'Aucun',
            quality_messages='; '.join(quality_messages),
            transcription=transcription,
            reference_text=reference_text,
            retrieved_context=retrieved_context
        )
        logger.info(f"Gemini prompt: {gemini_prompt}")

        gemini_headers = {"Content-Type": "application/json"}
        gemini_data = {
            "contents": [{"parts": [{"text": gemini_prompt}]}],
            "generationConfig": {"maxOutputTokens": 100, "temperature": 0.5}
        }
        gemini_response = requests.post(
            f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={gemini_api_key}",
            headers=gemini_headers,
            json=gemini_data
        )
        gemini_feedback = gemini_response.json()['candidates'][0]['content']['parts'][0]['text'].strip() if gemini_response.status_code == 200 else "Erreur Gemini."

        llama_prompt = """
Vous êtes un tuteur de français pour des élèves du primaire (niveau {level}). Donnez un retour CONCIS (<80 mots), POSITIF, et MOTIVANT, adapté aux enfants. Utilisez UNIQUEMENT les données et le contexte fournis.

Incluez :
- Points forts (prononciation, vitesse, compréhension).
- Conseils généraux pour s'améliorer, basés sur le contexte.
- Commentaire sur la qualité audio.

Données :
- Prononciation : {pronunciation_score}/100
- Compréhension : {text_accuracy}/100
- Vitesse : {words_per_minute} mots/min
- Mots mal prononcés : {mispronounced}
- Mots manquants : {missed_words}
- Mots supplémentaires : {extra_words}
- Qualité audio : {quality_messages}
Contexte :
{retrieved_context}
"""
        llama_prompt = llama_prompt.format(
            level=level,
            pronunciation_score=pronunciation_score,
            text_accuracy=text_accuracy,
            words_per_minute=words_per_minute,
            mispronounced=', '.join(mispronounced) or 'Aucun',
            missed_words=', '.join(missed_words) or 'Aucun',
            extra_words=', '.join(extra_words) or 'Aucun',
            quality_messages='; '.join(quality_messages),
            retrieved_context=retrieved_context
        )
        logger.info(f"LLaMA prompt: {llama_prompt}")

        llama_headers = {
            "Authorization": f"Bearer {hf_api_key}",
            "Content-Type": "application/json"
        }
        llama_data = {
            "inputs": llama_prompt,
            "parameters": {"max_new_tokens": 100, "temperature": 0.5}
        }
        llama_response = requests.post(
            "https://api-inference.huggingface.co/models/meta-llama/Llama-3.2-3B-Instruct",
            headers=llama_headers,
            json=llama_data
        )
        llama_feedback = llama_response.json()[0]['generated_text'].replace(llama_prompt, '').strip() if llama_response.status_code == 200 else "Erreur LLaMA."

        logger.info(f"Gemini feedback: {gemini_feedback}")
        logger.info(f"LLaMA feedback: {llama_feedback}")

        combined_feedback = f"**Gemini**: {gemini_feedback}\n**LLaMA**: {llama_feedback}"
        return {
            'feedback': combined_feedback,
            'error_details': {
                'mispronounced': [f"Mot {m}" for m in mispronounced] if mispronounced != ['Aucun'] else [],
                'missed': missed_words,
                'extra': extra_words
            }
        }
    except Exception as e:
        logger.error(f"Error generating feedback: {e}")
        return generate_fallback_feedback(pronunciation_score, text_accuracy, words_per_minute, pronunciation_errors, word_errors)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/get_random_text')
def get_random_text():
    try:
        language = request.args.get('language', 'fr')
        level = request.args.get('level')
        difficulty = request.args.get('difficulty')
        if not level or not difficulty:
            return jsonify({"status": "error", "message": "Level and difficulty are required"}), 400
        matching_texts = [t for t in text_db if t['level'] == int(level) and t['difficulty'] == difficulty]
        if not matching_texts:
            return jsonify({"status": "error", "message": "No matching text found"}), 404
        import random
        selected_text = random.choice(matching_texts)
        return jsonify({
            "status": "success",
            "text": {
                "id": selected_text['id'],
                "content": selected_text['text'],
                "level": selected_text['level'],
                "difficulty": selected_text['difficulty']
            }
        })
    except Exception as e:
        logger.error(f"Error fetching random text: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/text-to-speech', methods=['POST'])
def text_to_speech():
    try:
        data = request.json
        text = data.get('text')
        language = data.get('language', 'fr')
        if not text:
            return jsonify({"status": "error", "message": "Text is required"}), 400
        audio_path = os.path.join(AUDIO_OUTPUT_FOLDER, f"{uuid.uuid4()}.mp3")
        tts = gTTS(text=text, lang=language)
        tts.save(audio_path)
        audio_url = f"/{audio_path}"
        return jsonify({"status": "success", "audio_url": audio_url})
    except Exception as e:
        logger.error(f"Error generating TTS: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/evaluate', methods=['POST'])
def evaluate():
    try:
        if 'audio' not in request.files or 'level' not in request.form or 'difficulty' not in request.form:
            return jsonify({"status": "error", "message": "Missing audio, level, or difficulty"}), 400

        audio_file = request.files['audio']
        level = request.form['level']
        difficulty = request.form['difficulty']
        text_id = request.form.get('text_id', None)

        if not audio_file.filename.endswith(('.wav', '.mp3', '.webm', '.mp4')):
            return jsonify({"status": "error", "message": "Invalid audio format. Use WAV, MP3, WEBM, or MP4."}), 400

        temp_audio_path = os.path.join(UPLOAD_FOLDER, f"{uuid.uuid4()}.{audio_file.filename.split('.')[-1]}")
        audio_file.save(temp_audio_path)
        logger.info(f"Saved uploaded audio to {temp_audio_path}")

        quality_info = check_audio_quality(temp_audio_path)
        if not quality_info["is_valid"]:
            os.remove(temp_audio_path)
            return jsonify({"status": "error", "message": quality_info["messages"]}), 400

        text_entry = None
        if text_id:
            text_entry = next((item for item in text_db if item['id'] == text_id), None)
        if not text_entry:
            text_entry = next((item for item in text_db if item['level'] == int(level) and t['difficulty'] == difficulty), None)
        if not text_entry:
            os.remove(temp_audio_path)
            return jsonify({"status": "error", "message": "No matching text found"}), 404

        reference_text = text_entry['text']
        reference_audio_path = os.path.join(REFERENCE_AUDIO, f"{text_entry['id']}.mp3")
        if not os.path.exists(reference_audio_path):
            generate_reference_audio(reference_text, reference_audio_path)

        transcription = transcribe_audio(temp_audio_path)
        pronunciation_score, pronunciation_errors = analyze_pronunciation(reference_audio_path, temp_audio_path)
        text_accuracy, word_errors = analyze_text_accuracy(reference_text, transcription)
        words_per_minute = calculate_reading_speed(temp_audio_path, transcription)
        overall_score = (pronunciation_score + text_accuracy) / 2

        feedback = generate_grok_feedback(
            pronunciation_score, text_accuracy, words_per_minute, pronunciation_errors,
            transcription, reference_text, quality_info, word_errors, level
        )

        os.remove(temp_audio_path)

        return jsonify({
            "status": "success",
            "evaluation": {
                "pronunciation_accuracy": round(pronunciation_score, 1),
                "comprehension": round(text_accuracy, 1),
                "reading_speed": round(words_per_minute, 1),
                "score": round(overall_score, 1)
            },
            "llm_details": {
                "feedback": feedback['feedback'],
                "errors": feedback['error_details']
            },
            "transcription": transcription
        })
    except Exception as e:
        logger.error(f"Error in evaluation: {e}")
        if os.path.exists(temp_audio_path):
            os.remove(temp_audio_path)
        return jsonify({"status": "error", "message": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
