import os
import requests
import librosa
import numpy as np
from Levenshtein import distance as levenshtein_distance
from scipy.spatial.distance import euclidean
from dtw import dtw
import random
import logging

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def evaluate_reading(audio_path, reference_text, language='fr'):
    try:
        # Get OpenAI API key
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            logger.error("OPENAI_API_KEY not set in environment")
            raise ValueError("OPENAI_API_KEY not set")

        # Transcribe audio using OpenAI Whisper API
        url = "https://api.openai.com/v1/audio/transcriptions"
        headers = {
            "Authorization": f"Bearer {api_key}",
        }
        files = {
            "file": open(audio_path, 'rb'),
            "model": (None, "whisper-1"),
            "language": (None, 'fr' if language == 'fr' else 'ar')
        }

        logger.debug("Sending request to OpenAI Whisper API")
        response = requests.post(url, headers=headers, files=files)
        
        if response.status_code != 200:
            logger.error(f"Whisper API error: {response.status_code} - {response.text}")
            raise Exception(f"Whisper API error: {response.status_code} - {response.text}")

        transcription = response.json().get('text', '').strip().lower()
        logger.debug(f"Transcription: {transcription}")

        # Load audio for acoustic analysis
        y, sr = librosa.load(audio_path)
        
        # Acoustic Analysis: MFCCs
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        mfccs_ref = mfccs  # Placeholder: Compare with reference audio if available
        
        # Dynamic Time Warping for pronunciation alignment
        dtw_distance, _ = dtw(mfccs.T, mfccs_ref.T, dist=euclidean)
        pronunciation_accuracy = max(0, 100 - dtw_distance / 10)  # Simplified scoring
        
        # Additional Audio Features
        spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
        rms_energy = np.mean(librosa.feature.rms(y=y))
        zero_crossing_rate = np.mean(librosa.feature.zero_crossing_rate(y=y))
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        
        # Reading Speed (Words per Minute)
        audio_duration = len(y) / sr
        word_count = len(transcription.split())
        wpm = (word_count / audio_duration) * 60 if audio_duration > 0 else 0
        
        # Text Analysis: Levenshtein Distance
        reference_text_clean = reference_text.lower().strip()
        lev_distance = levenshtein_distance(transcription, reference_text_clean)
        max_len = max(len(transcription), len(reference_text_clean))
        text_accuracy = max(0, 100 - (lev_distance / max_len * 100)) if max_len > 0 else 0
        
        # Comprehension (Placeholder: Requires LLM-based analysis)
        comprehension_score = random.randint(80, 100)  # Mock score
        
        # Overall Score
        overall_score = (pronunciation_accuracy * 0.4 + text_accuracy * 0.4 + comprehension_score * 0.2)
        
        # Generate Feedback (Mock: Replace with LLaMA API if needed)
        pronunciation_feedback = f"Prononciation: {'Bonne' if pronunciation_accuracy > 80 else 'À améliorer'}. Spectral centroid: {spectral_centroid:.2f}, RMS: {rms_energy:.2f}, Tempo: {tempo:.2f}."
        comprehension_feedback = "Compréhension: Bonne compréhension générale."
        
        logger.debug(f"Evaluation results: pronunciation={pronunciation_accuracy}, wpm={wpm}, comprehension={comprehension_score}")
        
        return {
            'scores': {
                'pronunciation_accuracy': round(pronunciation_accuracy),
                'reading_speed': 'Optimal' if 120 <= wpm <= 160 else 'Trop rapide' if wpm > 160 else 'Trop lent',
                'comprehension': comprehension_score,
                'score': round(overall_score)
            },
            'wpm': round(wpm),
            'feedback': {
                'pronunciation': {'feedback': pronunciation_feedback},
                'comprehension': {'feedback': comprehension_feedback}
            }
        }
    except Exception as e:
        logger.error(f"Error in evaluate_reading: {str(e)}")
        raise