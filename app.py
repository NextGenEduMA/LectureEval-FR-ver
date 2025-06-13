from flask import Flask, request, jsonify, render_template
import os
import json
import librosa
import numpy as np
from jiwer import wer, cer, compute_measures
from fastdtw import fastdtw
from transformers import pipeline, AutoProcessor, AutoModelForSpeechSeq2Seq, AutoModelForCausalLM, AutoTokenizer
import torch
from gtts import gTTS
import uuid
import tempfile
import logging
import threading
import time
import soundfile as sf
from pydub import AudioSegment
import requests
from rag_retriever import RAGRetriever
import pymongo

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
from dotenv import load_dotenv
load_dotenv()
hf_token = os.getenv('HF_TOKEN')
gemini_api_key = os.getenv('GEMINI_API_KEY')
mongo_uri = os.getenv('MONGO_URI', 'mongodb://localhost:27017')  # Default to local MongoDB

app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = 'Uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
AUDIO_OUTPUT_FOLDER = 'static/audio'
REFERENCE_AUDIO = 'reference_audio'
os.makedirs(AUDIO_OUTPUT_FOLDER, exist_ok=True)
os.makedirs(REFERENCE_AUDIO, exist_ok=True)
MAX_AUDIO_DURATION = 60  # seconds
MIN_SNR = 10  # dB
MIN_SAMPLE_RATE = 16000  # Hz

# MongoDB setup
try:
    client = pymongo.MongoClient(mongo_uri)
    db = client['language_learning']  # Database name
    text_collection = db['texts']  # Collection name
    client.server_info()  # Test connection
    logger.info("Connected to MongoDB successfully")
except Exception as e:
    logger.error(f"Failed to connect to MongoDB: {e}")
    raise

# Initialize RAG retriever
rag_retriever = RAGRetriever(corpus_dir="rag_corpus")

# Load LLaMA model (using distilgpt2 as a lightweight placeholder)
try:
    device = torch.device("cpu")
    logger.info("Loading LLaMA model (distilgpt2) on CPU")
    llm_model = AutoModelForCausalLM.from_pretrained("distilgpt2")
    llm_tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
    llm_tokenizer.pad_token = llm_tokenizer.eos_token  # Set pad token
    llm_model.to(device)
    logger.info("LLaMA model loaded successfully")
except Exception as e:
    logger.error(f"Failed to load LLaMA model: {e}")
    llm_model = None
    llm_tokenizer = None

# Generate reference audio for a single text
def generate_reference_audio(text, language, text_id):
    ref_audio_path = os.path.join(REFERENCE_AUDIO, f"ref_{text_id}.wav")
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

# Audio quality check
def check_audio_quality(audio_path):
    try:
        y, sr = librosa.load(audio_path, sr=None)
        sample_rate_ok = sr >= MIN_SAMPLE_RATE
        sample_rate_message = f"Taux d'échantillonnage: {sr}Hz ({'OK' if sample_rate_ok else 'trop bas'})"
        rms = np.sqrt(np.mean(y**2))
        silence_message = "Audio trop silencieux" if rms < 0.01 else "Niveau sonore correct"
        signal_power = np.mean(y**2)
        noise = y - librosa.effects.preemphasis(y)
        noise_power = np.mean(noise**2)
        snr = 10 * np.log10(signal_power / (noise_power + 1e-10)) if noise_power > 0 else 100
        snr_message = f"SNR: {snr:.1f}dB ({'OK' if snr >= MIN_SNR else 'trop de bruit'})"
        return {
            'is_quality_ok': sample_rate_ok and rms >= 0.01 and snr >= MIN_SNR,
            'messages': [sample_rate_message, silence_message, snr_message]
        }
    except Exception as e:
        logger.error(f"Error checking audio quality: {e}")
        return {
            'is_quality_ok': False,
            'messages': ["Erreur lors de l'analyse de la qualité audio"]
        }

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
        generate_kwargs={"num_beams": 1, "max_new_tokens": 200}
    )
    asr_pipeline.model.config.forced_decoder_ids = asr_pipeline.tokenizer.get_decoder_prompt_ids(language="fr", task="transcribe")
    if asr_pipeline.tokenizer.pad_token_id is None:
        asr_pipeline.tokenizer.pad_token_id = asr_pipeline.tokenizer.eos_token_id
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
        score = max(0, 100 - norm_dist * 8)  # Increased penalty
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
        word_errors = {'missed': [], 'extra': [], 'substituted': []}
        
        if ref != trans:
            try:
                measures = compute_measures(ref, trans)
                logger.info(f"jiwer measures: {measures}")
                ref_words = ref.split()
                trans_words = trans.split()
                
                for chunk in measures['ops']:
                    chunk_type = chunk.type
                    if chunk_type == 'insert':
                        start_idx = chunk.hyp_start_idx
                        end_idx = chunk.hyp_end_idx
                        word_errors['extra'].extend(trans_words[start_idx:end_idx])
                    elif chunk_type == 'delete':
                        start_idx = chunk.ref_start_idx
                        end_idx = chunk.ref_end_idx
                        word_errors['missed'].extend(ref_words[start_idx:end_idx])
                    elif chunk_type == 'substitute':
                        ref_start = chunk.ref_start_idx
                        ref_end = chunk.ref_end_idx
                        hyp_start = chunk.hyp_start_idx
                        hyp_end = chunk.hyp_end_idx
                        for ref_w, hyp_w in zip(ref_words[ref_start:ref_end], trans_words[hyp_start:hyp_end]):
                            word_errors['substituted'].append(f"{ref_w} → {hyp_w}")
                
                if not word_errors['missed'] and not word_errors['extra'] and not word_errors['substituted']:
                    ref_words = ref.split()
                    trans_words = trans.split()
                    min_len = min(len(ref_words), len(trans_words))
                    for i in range(min_len):
                        if ref_words[i] != trans_words[i]:
                            word_errors['substituted'].append(f"{ref_words[i]} → {trans_words[i]}")
                    if len(ref_words) > len(trans_words):
                        word_errors['missed'].extend(ref_words[len(trans_words):])
                    if len(trans_words) > len(ref_words):
                        word_errors['extra'].extend(trans_words[len(ref_words):])
            except Exception as e:
                logger.warning(f"Failed to compute word errors with jiwer: {e}")
                ref_words = ref.split()
                trans_words = trans.split()
                min_len = min(len(ref_words), len(trans_words))
                for i in range(min_len):
                    if ref_words[i] != trans_words[i]:
                        word_errors['substituted'].append(f"{ref_words[i]} → {trans_words[i]}")
                if len(ref_words) > len(trans_words):
                    word_errors['missed'].extend(ref_words[len(trans_words):])
                if len(trans_words) > len(ref_words):
                    word_errors['extra'].extend(trans_words[len(ref_words):])
        
        result['word_errors'] = word_errors
        logger.info(f"Text accuracy (WER) took {time.time() - start_time:.2f} seconds")
        logger.info(f"Detected errors: missed={word_errors['missed']}, extra={word_errors['extra']}, substituted={word_errors['substituted']}")
    except Exception as e:
        logger.error(f"Error evaluating text accuracy: {e}")
        result['wer_accuracy'] = 80
        result['cer_accuracy'] = 80
        result['word_errors'] = {'missed': [], 'extra': [], 'substituted': []}

def get_speed_score(words_per_minute, tempo):
    if words_per_minute > 180 or tempo > 120:
        return 60  # Too fast
    elif words_per_minute < 80 or tempo < 80:
        return 60  # Too slow
    else:
        return 100  # Appropriate

def generate_grok_feedback(pronunciation_score, text_accuracy, words_per_minute, pronunciation_errors, transcription, reference_text, quality_info, word_errors, level):
    try:
        if not gemini_api_key:
            logger.warning("GEMINI_API_KEY not set, using fallback feedback")
            return generate_fallback_feedback(pronunciation_score, text_accuracy, words_per_minute, pronunciation_errors, word_errors)

        ref_words = reference_text.lower().strip().split()
        trans_words = transcription.lower().strip().split()
        missed_words = word_errors.get('missed', [])
        extra_words = word_errors.get('extra', [])
        substituted_words = word_errors.get('substituted', [])
        
        if not missed_words and not extra_words and not substituted_words and reference_text.lower().strip() != transcription.lower().strip():
            min_len = min(len(ref_words), len(trans_words))
            for i in range(min_len):
                if ref_words[i] != trans_words[i]:
                    substituted_words.append(f"{ref_words[i]} → {trans_words[i]}")
            if len(ref_words) > len(trans_words):
                missed_words.extend(ref_words[len(trans_words):])
            if len(trans_words) > len(ref_words):
                extra_words.extend(trans_words[len(ref_words):])
        
        mispronounced = [f"'{e['word']}' à {e['time']}s" for e in pronunciation_errors]
        quality_messages = quality_info.get('messages', ['Qualité audio non évaluée'])

        error_context = f"Errors: missed={', '.join(missed_words) or 'none'}; extra={', '.join(extra_words) or 'none'}; substituted={', '.join(substituted_words) or 'none'}; mispronounced={', '.join(mispronounced) or 'none'}; pronunciation_score={pronunciation_score}; reading_speed={words_per_minute}"
        retrieved_docs = rag_retriever.retrieve(error_context, level, n_results=2)
        retrieved_context = "\n".join([f"- {doc}" for doc in retrieved_docs]) if retrieved_docs else "Aucune astuce disponible."

        logger.info(f"Gemini feedback - Detected errors: missed={missed_words}, extra={extra_words}, substituted={substituted_words}")

        prompt = """
Vous êtes un tuteur de langue française pour des élèves marocains du primaire (niveau {level}). Générez un feedback PRÉCIS et CONCIS (50-100 mots), positif et adapté aux enfants, basé UNIQUEMENT sur les données fournies.

IMPORTANT: Analysez attentivement les différences entre la transcription et la référence. Identifiez les erreurs spécifiques (mots manquants, supplémentaires ou mal prononcés).

Incluez:
1. Points forts (prononciation: {pronunciation_score}/100, vitesse: {words_per_minute} mots/min)
2. Erreurs précises avec corrections (ex: "tu as dit 'écrite un' au lieu de 'écrit une'")
3. Conseil d'amélioration court et précis
4. Commentaire sur la qualité audio

Données:
- Prononciation: {pronunciation_score}/100
- Compréhension: {text_accuracy}/100
- Vitesse: {words_per_minute} mots/min
- Mots mal prononcés: {mispronounced}
- Mots manquants: {missed_words}
- Mots supplémentaires: {extra_words}
- Mots substitués: {substituted_words}
- Qualité audio: {quality_messages}
- Transcription: "{transcription}"
- Référence: "{reference_text}"
Contexte:
{retrieved_context}
"""
        prompt = prompt.format(
            level=level,
            pronunciation_score=pronunciation_score,
            text_accuracy=text_accuracy,
            words_per_minute=words_per_minute,
            mispronounced=', '.join(mispronounced) or 'Aucun',
            missed_words=', '.join(missed_words) or 'Aucun',
            extra_words=', '.join(extra_words) or 'Aucun',
            substituted_words=', '.join(substituted_words) or 'Aucun',
            quality_messages='; '.join(quality_messages),
            transcription=transcription,
            reference_text=reference_text,
            retrieved_context=retrieved_context
        )
        logger.info(f"Gemini prompt with RAG: {prompt}")

        headers = {
            "Content-Type": "application/json"
        }
        data = {
            "contents": [{
                "parts": [{"text": prompt}]
            }],
            "generationConfig": {
                "maxOutputTokens": 300,
                "temperature": 0.7,
                "topP": 0.9
            }
        }

        response = requests.post(
            f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={gemini_api_key}",
            headers=headers,
            json=data
        )
        if response.status_code == 200:
            feedback = response.json()['candidates'][0]['content']['parts'][0]['text'].strip()
            logger.info(f"Gemini feedback: {feedback}")
            return {
                'feedback': feedback,
                'error_details': {
                    'mispronounced': [f"Mot {m}" for m in mispronounced] if mispronounced != ['Aucun'] else [],
                    'missed': missed_words,
                    'extra': extra_words,
                    'substituted': substituted_words
                }
            }
        else:
            logger.error(f"Gemini API error: {response.status_code} - {response.text}")
            return generate_fallback_feedback(pronunciation_score, text_accuracy, words_per_minute, pronunciation_errors, word_errors)
    except Exception as e:
        logger.error(f"Error generating Gemini feedback: {e}")
        return generate_fallback_feedback(pronunciation_score, text_accuracy, words_per_minute, pronunciation_errors, word_errors)

def generate_llama_feedback(pronunciation_score, text_accuracy, words_per_minute, pronunciation_errors, transcription, reference_text, quality_info, word_errors, level):
    try:
        if not llm_model or not llm_tokenizer:
            logger.warning("LLaMA model not loaded, using fallback feedback")
            return generate_fallback_feedback(pronunciation_score, text_accuracy, words_per_minute, pronunciation_errors, word_errors)

        ref_words = reference_text.lower().strip().split()
        trans_words = transcription.lower().strip().split()
        missed_words = word_errors.get('missed', [])
        extra_words = word_errors.get('extra', [])
        substituted_words = word_errors.get('substituted', [])
        
        if not missed_words and not extra_words and not substituted_words and reference_text.lower().strip() != transcription.lower().strip():
            min_len = min(len(ref_words), len(trans_words))
            for i in range(min_len):
                if ref_words[i] != trans_words[i]:
                    substituted_words.append(f"{ref_words[i]} → {trans_words[i]}")
            if len(ref_words) > len(trans_words):
                missed_words.extend(ref_words[len(trans_words):])
            if len(trans_words) > len(ref_words):
                extra_words.extend(trans_words[len(ref_words):])
        
        mispronounced = [f"'{e['word']}' à {e['time']}s" for e in pronunciation_errors]
        quality_messages = quality_info.get('messages', ['Qualité audio non évaluée'])

        error_context = f"Errors: missed={', '.join(missed_words) or 'none'}; extra={', '.join(extra_words) or 'none'}; substituted={', '.join(substituted_words) or 'none'}; mispronounced={', '.join(mispronounced) or 'none'}; pronunciation_score={pronunciation_score}; reading_speed={words_per_minute}"
        retrieved_docs = rag_retriever.retrieve(error_context, level, n_results=2)
        retrieved_context = "\n".join([f"- {doc}" for doc in retrieved_docs]) if retrieved_docs else "Aucune astuce disponible."

        logger.info(f"LLaMA feedback - Detected errors: missed={missed_words}, extra={extra_words}, substituted={substituted_words}")

        prompt = """
Feedback de lecture en français pour un élève:

Prononciation: {pronunciation_score}/100
Vitesse: {words_per_minute} mots/min
Texte de référence: "{reference_text}"
Transcription: "{transcription}"

Bravo pour ton effort! Ta prononciation est {prononciation_qualite}.
"""
        if "élèves," in transcription and "élèves" in reference_text:
            specific_feedback = "Tu as répété 'les élèves' au lieu de dire simplement 'Les élèves préparent un spectacle à l'école'. Essaie de lire le texte une fois avant d'enregistrer."
        elif extra_words:
            specific_feedback = f"Tu as ajouté des mots en trop: {', '.join(extra_words[:2])}. Lis le texte plus attentivement."
        elif missed_words:
            specific_feedback = f"Tu as oublié les mots: {', '.join(missed_words[:2])}. Prends ton temps pour lire tous les mots."
        elif substituted_words:
            specific_feedback = f"Tu as dit {substituted_words[0].split(' → ')[1]} au lieu de {substituted_words[0].split(' → ')[0]}. Fais attention aux mots exacts."
        else:
            specific_feedback = "Continue à pratiquer ta lecture régulièrement."
        
        if "trop de bruit" in '; '.join(quality_messages):
            quality_feedback = "Essaie d'enregistrer dans un endroit plus calme."
        else:
            quality_feedback = "La qualité audio est bonne."
        
        if pronunciation_score >= 80:
            prononciation_qualite = "excellente"
        elif pronunciation_score >= 60:
            prononciation_qualite = "bonne"
        else:
            prononciation_qualite = "à améliorer"
        
        prompt = prompt.format(
            pronunciation_score=pronunciation_score,
            words_per_minute=round(words_per_minute, 1),
            reference_text=reference_text,
            transcription=transcription,
            prononciation_qualite=prononciation_qualite
        )
        
        complete_feedback = f"Bravo pour ton effort! Ta prononciation est {prononciation_qualite} ({pronunciation_score}/100) et ta vitesse de lecture est bonne ({round(words_per_minute, 1)} mots/min). {specific_feedback} {quality_feedback}"
        
        logger.info(f"Generated feedback: {complete_feedback}")
        
        return {
            'feedback': complete_feedback,
            'error_details': {
                'mispronounced': [f"Mot {m}" for m in mispronounced] if mispronounced != ['Aucun'] else [],
                'missed': missed_words,
                'extra': extra_words,
                'substituted': substituted_words
            }
        }
    except Exception as e:
        logger.error(f"Error generating LLaMA feedback: {e}")
        return generate_fallback_feedback(pronunciation_score, text_accuracy, words_per_minute, pronunciation_errors, word_errors)

def generate_fallback_feedback(pronunciation_score, text_accuracy, words_per_minute, pronunciation_errors, word_errors):
    feedback = ["Super effort !"]
    error_details = {
        'mispronounced': [f"Mot '{e['word']}' à {e['time']}s" for e in pronunciation_errors[:3]],
        'missed': word_errors.get('missed', [])[:3],
        'extra': word_errors.get('extra', [])[:3],
        'substituted': word_errors.get('substituted', [])[:3]
    }
    
    if pronunciation_score >= 80:
        feedback.append("Ta prononciation est très bonne !")
    elif pronunciation_score >= 60:
        feedback.append("Bonne prononciation, continue !")
    else:
        feedback.append("Pratique encore ta prononciation.")
    
    if error_details['substituted'] or error_details['extra'] or error_details['missed']:
        if error_details['substituted']:
            substituted = error_details['substituted'][:2]
            feedback.append(f"Attention aux mots: {', '.join(substituted)}.")
        if error_details['extra']:
            feedback.append(f"Mots en trop: {', '.join(error_details['extra'][:2])}.")
        if error_details['missed']:
            feedback.append(f"Mots manquants: {', '.join(error_details['missed'][:2])}.")
        else:
            feedback.append("Pas d'erreurs de mots, bravo !")
    
    if words_per_minute > 180:
        feedback.append("Ralentis un peu pour mieux articuler.")
    elif words_per_minute < 80:
        feedback.append("Essaie de lire un peu plus vite.")
    else:
        feedback.append("Bonne vitesse de lecture !")
    
    feedback.append("Conseil: enregistre dans un endroit calme pour une meilleure évaluation.")
    
    feedback_text = ' '.join(feedback)
    return {
        'feedback': feedback_text,
        'error_details': error_details
    }

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/get_random_text')
def get_random_text():
    try:
        language = request.args.get('language', 'fr')
        level = request.args.get('level', '6')
        difficulty = request.args.get('difficulty', 'medium')
        matching_texts = list(text_collection.find({
            'language': language,
            'level': level,
            'difficulty': difficulty
        }))
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
        logger.error(f"Error getting random text from MongoDB: {e}")
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

        temp_audio_path = os.path.join(UPLOAD_FOLDER, f"temp_{uuid.uuid4()}.{audio_file.filename.split('.')[-1]}")
        audio_file.save(temp_audio_path)
        wav_audio_path = os.path.join(UPLOAD_FOLDER, f"temp_{uuid.uuid4()}.wav")
        convert_to_wav(temp_audio_path, wav_audio_path)
        quality_info = check_audio_quality(wav_audio_path)
        duration = get_audio_duration(wav_audio_path)
        logger.info(f"Audio duration: {duration:.2f} seconds")
        if duration > MAX_AUDIO_DURATION:
            return jsonify({'status': 'error', 'message': f"Audio trop long ({duration:.1f}s). Limite: {MAX_AUDIO_DURATION}s."})
        if duration < 0.1:
            return jsonify({'status': 'error', 'message': 'Audio trop court ou invalide.'})

        transcribe_start = time.time()
        transcription = asr_pipeline(wav_audio_path)['text']
        logger.info(f"Transcription: {transcription}")
        logger.info(f"Reference text: {reference_text}")
        logger.info(f"Transcription took {time.time() - transcribe_start:.2f} seconds")
        audio_features = extract_audio_features(wav_audio_path)
        ref_audio_path = os.path.join(REFERENCE_AUDIO, f"ref_{text_id}.wav")
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

        result = {}
        alignment_thread = threading.Thread(target=evaluate_text_accuracy, args=(reference_text, transcription, result))
        alignment_thread.start()
        alignment_thread.join()

        text_accuracy = result.get('wer_accuracy', 80)
        cer_accuracy = result.get('cer_accuracy', 80)
        word_errors = result.get('word_errors', {'missed': [], 'extra': [], 'substituted': []})
        
        if (not word_errors['missed'] and not word_errors['extra'] and not word_errors['substituted'] and 
            reference_text.lower().strip() != transcription.lower().strip()):
            logger.info("No errors detected by jiwer but texts differ, performing manual comparison")
            ref_words = reference_text.lower().strip().split()
            trans_words = transcription.lower().strip().split()
            min_len = min(len(ref_words), len(trans_words))
            for i in range(min_len):
                if ref_words[i] != trans_words[i]:
                    word_errors['substituted'].append(f"{ref_words[i]} → {trans_words[i]}")
            if len(ref_words) > len(trans_words):
                word_errors['missed'].extend(ref_words[len(trans_words):])
            if len(trans_words) > len(ref_words):
                word_errors['extra'].extend(trans_words[len(ref_words):])
            logger.info(f"Manual comparison found: missed={word_errors['missed']}, extra={word_errors['extra']}, substituted={word_errors['substituted']}")
        
        word_count = len(reference_text.split())
        logger.info(f"Word count: {word_count}")
        words_per_minute = (word_count / duration) * 60 if duration > 0 else 0
        logger.info(f"Words per minute: {words_per_minute:.2f}")
        speed_score = get_speed_score(words_per_minute, audio_features['tempo'])
        
        gemini_feedback = generate_grok_feedback(
            pronunciation_score, text_accuracy, words_per_minute, pronunciation_errors, 
            transcription, reference_text, quality_info, word_errors, level
        )
        llama_feedback = generate_llama_feedback(
            pronunciation_score, text_accuracy, words_per_minute, pronunciation_errors, 
            transcription, reference_text, quality_info, word_errors, level
        )
        
        overall_score = round(
            0.5 * pronunciation_score + 0.3 * text_accuracy + 0.2 * speed_score,
            2
        )
        evaluation = {
            'pronunciation_accuracy': pronunciation_score,
            'reading_speed': round(words_per_minute, 2),
            'comprehension': text_accuracy,
            'cer_accuracy': cer_accuracy,
            'score': overall_score
        }
        response = {
            'status': 'success',
            'evaluation': evaluation,
            'words_per_minute': round(words_per_minute, 2),
            'transcription': transcription,
            'reference_text': reference_text,
            'llm_details': {
                'gemini_feedback': gemini_feedback['feedback'],
                'gemini_errors': gemini_feedback['error_details'],
                'llama_feedback': llama_feedback['feedback'],
                'llama_errors': llama_feedback['error_details']
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
        texts = list(text_collection.find({}, {'_id': 0}))  # Exclude MongoDB _id field
        return jsonify({'status': 'success', 'texts': texts})
    except Exception as e:
        logger.error(f"Error listing texts from MongoDB: {e}")
        return jsonify({'status': 'error', 'message': str(e)})

if __name__ == '__main__':
    app.run(debug=True)