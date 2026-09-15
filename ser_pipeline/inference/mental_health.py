"""Heuristic mental-health indicator scoring, built on top of raw emotion
probabilities. These are hand-set weights/thresholds, not learned from
labeled mental-health data — treat outputs as a coarse screening signal,
not a diagnosis."""
import os
from datetime import datetime

import numpy as np

from inference.predictor import predict_emotion

MH_PROFILES = {
    'Depression': {
        'description': 'Low mood, anhedonia, emotional blunting',
        'weights': {'Sad': 0.50, 'Neutral': 0.20, 'Fear': 0.15, 'Disgust': 0.10,
                    'Happy': -0.50, 'Angry': -0.05},
        'threshold_low': 0.30, 'threshold_high': 0.55, 'color': '#3498db'},
    'Anxiety': {
        'description': 'Excessive worry, fear, hyperarousal',
        'weights': {'Fear': 0.55, 'Angry': 0.25, 'Disgust': 0.10, 'Neutral': -0.10,
                    'Happy': -0.30, 'Sad': 0.05},
        'threshold_low': 0.25, 'threshold_high': 0.50, 'color': '#e74c3c'},
    'Stress': {
        'description': 'Emotional tension, irritability, pressure',
        'weights': {'Angry': 0.55, 'Fear': 0.20, 'Disgust': 0.20, 'Sad': 0.05,
                    'Happy': -0.30, 'Neutral': -0.10},
        'threshold_low': 0.28, 'threshold_high': 0.52, 'color': '#e67e22'},
    'Emotional_Wellbeing': {
        'description': 'Positive affect, resilience, stability',
        'weights': {'Happy': 0.60, 'Neutral': 0.15, 'Sad': -0.30, 'Fear': -0.25,
                    'Angry': -0.25, 'Disgust': -0.15},
        'threshold_low': 0.35, 'threshold_high': 0.60, 'color': '#2ecc71'},
}
MH_SCORE_OFFSET = {'Depression': 0.25, 'Anxiety': 0.20, 'Stress': 0.20,
                    'Emotional_Wellbeing': 0.25}
MH_SCORE_SCALE = {'Depression': 0.75, 'Anxiety': 0.70, 'Stress': 0.70,
                   'Emotional_Wellbeing': 0.75}


def emotion_probs_to_mh_scores(emotion_probs: dict) -> dict:
    scores = {}
    for condition, profile in MH_PROFILES.items():
        raw = sum(emotion_probs.get(em, 0.0) * w for em, w in profile['weights'].items())
        score = float(np.clip(
            (raw + MH_SCORE_OFFSET[condition]) / MH_SCORE_SCALE[condition], 0.0, 1.0))
        scores[condition] = round(score, 4)
    return scores


def score_to_risk_level(score: float, condition: str) -> str:
    lo, hi = MH_PROFILES[condition]['threshold_low'], MH_PROFILES[condition]['threshold_high']
    if condition == 'Emotional_Wellbeing':
        return 'Good' if score >= hi else ('Moderate' if score >= lo else 'Low - Concern')
    return 'Low' if score <= lo else ('Moderate' if score <= hi else 'Elevated - Attention Needed')


def analyze_mental_health(audio_path: str, trained_model=None, session_id: str = None) -> dict:
    if session_id is None:
        session_id = datetime.now().strftime('%Y%m%d_%H%M%S')
    emotion_result = predict_emotion(audio_path, ema_model=trained_model)
    mh_scores = emotion_probs_to_mh_scores(emotion_result['class_probabilities'])
    mh_info, flags = {}, []
    for condition, score in mh_scores.items():
        risk = score_to_risk_level(score, condition)
        mh_info[condition] = {'score': score, 'risk_level': risk,
                               'description': MH_PROFILES[condition]['description']}
        if condition != 'Emotional_Wellbeing' and risk.startswith('Elevated'):
            flags.append(f'[!] {condition}: score={score:.2f} - {risk}')
        elif condition == 'Emotional_Wellbeing' and risk.startswith('Low'):
            flags.append(f'[!] {condition}: score={score:.2f} - {risk}')
    if not flags:
        flags.append('[OK] No concerning indicators detected')
    return {
        'session_id': session_id,
        'audio_file': os.path.basename(audio_path),
        'emotion': {'predicted': emotion_result['predicted_emotion'],
                    'confidence': emotion_result['confidence'],
                    'uncertainty': emotion_result['uncertainty'],
                    'probabilities': emotion_result['class_probabilities']},
        'mental_health': mh_info,
        'flags': flags,
    }
