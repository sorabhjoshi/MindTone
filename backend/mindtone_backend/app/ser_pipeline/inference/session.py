"""Aggregates mental-health indicators across a session of multiple
utterances from the same speaker (confidence-weighted average, trend
detection, mood-stability estimate)."""
from collections import defaultdict
from datetime import datetime

import numpy as np

from inference.mental_health import MH_PROFILES, analyze_mental_health, score_to_risk_level


class MentalHealthSession:
    def __init__(self, patient_id: str = 'patient_001', session_name: str = None):
        self.patient_id = patient_id
        self.session_name = session_name or datetime.now().strftime('session_%Y%m%d_%H%M')
        self.utterances = []

    def add_utterance(self, audio_path: str, trained_model=None) -> dict:
        utt_id = f'utt_{len(self.utterances) + 1:03d}'
        report = analyze_mental_health(audio_path, trained_model=trained_model,
                                        session_id=utt_id)
        self.utterances.append(report)
        return report

    def get_session_summary(self) -> dict:
        if not self.utterances:
            return {'error': 'No utterances recorded'}
        n = len(self.utterances)
        conf_weights = np.array([u['emotion']['confidence'] ** 2 for u in self.utterances])
        conf_weights = conf_weights / conf_weights.sum().clip(min=1e-9)

        emotion_acc = defaultdict(float)
        for w, utt in zip(conf_weights, self.utterances):
            for em, p in utt['emotion']['probabilities'].items():
                emotion_acc[em] += w * p
        avg_emotion_probs = {em: round(float(v), 4) for em, v in emotion_acc.items()}
        dominant_emotion = max(avg_emotion_probs, key=avg_emotion_probs.get)

        mh_acc = defaultdict(float)
        for w, utt in zip(conf_weights, self.utterances):
            for cond, info in utt['mental_health'].items():
                mh_acc[cond] += w * info['score']
        avg_mh_scores = {c: round(float(v), 4) for c, v in mh_acc.items()}
        risk_levels = {c: score_to_risk_level(s, c) for c, s in avg_mh_scores.items()}

        mh_variance = {}
        for cond in MH_PROFILES:
            vals = [u['mental_health'][cond]['score'] for u in self.utterances]
            mh_variance[cond] = round(float(np.std(vals)), 4)

        trends = {}
        if n >= 4:
            mid = n // 2
            for cond in MH_PROFILES:
                first = np.mean([u['mental_health'][cond]['score'] for u in self.utterances[:mid]])
                second = np.mean([u['mental_health'][cond]['score'] for u in self.utterances[mid:]])
                delta = second - first
                trends[cond] = {
                    'direction': 'Stable' if abs(delta) < 0.04
                    else ('Increasing' if delta > 0 else 'Decreasing'),
                    'delta': round(delta, 3)}
        else:
            trends = {c: {'direction': 'Insufficient data', 'delta': 0.0} for c in MH_PROFILES}

        entropy_vals = []
        for utt in self.utterances:
            probs = np.clip(list(utt['emotion']['probabilities'].values()), 1e-9, 1.0)
            entropy_vals.append(-np.sum(probs * np.log(probs)))
        std_entropy = float(np.std(entropy_vals))
        mood_stability = ('Stable' if std_entropy < 0.25
                           else ('Variable' if std_entropy < 0.55 else 'Highly Variable'))

        avg_confidence = float(np.mean([u['emotion']['confidence'] for u in self.utterances]))
        avg_uncertainty = float(np.mean([u['emotion']['uncertainty'] for u in self.utterances]))

        flags = []
        if avg_mh_scores['Depression'] > MH_PROFILES['Depression']['threshold_high']:
            flags.append('[!] DEPRESSION: Sustained elevated indicators')
        if avg_mh_scores['Anxiety'] > MH_PROFILES['Anxiety']['threshold_high']:
            flags.append('[!] ANXIETY: Sustained elevated indicators')
        if avg_mh_scores['Stress'] > MH_PROFILES['Stress']['threshold_high']:
            flags.append('[!] STRESS: Sustained elevated indicators')
        if avg_mh_scores['Emotional_Wellbeing'] < MH_PROFILES['Emotional_Wellbeing']['threshold_low']:
            flags.append('[!] WELLBEING: Consistently low positive affect')
        if avg_confidence < 0.35:
            flags.append(f'[!] LOW CONFIDENCE: mean={avg_confidence:.2f} - treat MH scores cautiously')
        if not flags:
            flags.append('[OK] Session indicators within normal range')

        return {
            'patient_id': self.patient_id,
            'session_name': self.session_name,
            'n_utterances': n,
            'dominant_emotion': dominant_emotion,
            'avg_emotion_probs': avg_emotion_probs,
            'avg_mh_scores': avg_mh_scores,
            'mh_variance': mh_variance,
            'risk_levels': risk_levels,
            'trends': trends,
            'mood_stability': mood_stability,
            'avg_confidence': round(avg_confidence, 4),
            'avg_uncertainty': round(avg_uncertainty, 4),
            'flags': flags,
        }
