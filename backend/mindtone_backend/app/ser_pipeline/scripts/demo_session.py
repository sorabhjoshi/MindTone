#!/usr/bin/env python3
"""Demo B — process several audio files as one session and print an
aggregated mental-health summary (trend, mood stability, dominant emotion).

Usage:
    python scripts/demo_session.py clip1.wav clip2.wav clip3.wav ...
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from inference.session import MentalHealthSession  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('clips', nargs='+', help='Audio file paths, in session order')
    p.add_argument('--patient-id', default='DEMO_001')
    args = p.parse_args()

    print('=' * 72)
    print(f'  SESSION-LEVEL MENTAL HEALTH MONITORING  ({len(args.clips)} utterances)')
    print('=' * 72)
    session = MentalHealthSession(patient_id=args.patient_id)
    print('  Processing utterances:')
    for clip in args.clips:
        utt = session.add_utterance(clip)
        mh = utt['mental_health']
        print(f"    Pred:{utt['emotion']['predicted']:8s}  "
              f"conf={utt['emotion']['confidence']:.2f}  "
              f"D={mh['Depression']['score']:.2f} "
              f"A={mh['Anxiety']['score']:.2f} "
              f"S={mh['Stress']['score']:.2f} "
              f"W={mh['Emotional_Wellbeing']['score']:.2f}  "
              f"unc={utt['emotion']['uncertainty']:.2f}")

    summary = session.get_session_summary()
    print()
    print('  SESSION SUMMARY:')
    print(f"    Patient        : {summary['patient_id']}")
    print(f"    Utterances     : {summary['n_utterances']}")
    print(f"    Dominant Emo   : {summary['dominant_emotion']}")
    print(f"    Mood stability : {summary['mood_stability']}")
    print(f"    Avg confidence : {summary['avg_confidence']:.3f}")
    print()
    print('  Average MH Scores:')
    for cond, score in summary['avg_mh_scores'].items():
        risk = summary['risk_levels'][cond]
        trend = summary['trends'][cond]['direction']
        var = summary['mh_variance'].get(cond, 0.0)
        bar = '#' * int(score * 30)
        space = '.' * (30 - len(bar))
        print(f'    {cond:22s} {score:5.3f} +/-{var:.3f}  |{bar}{space}| {risk}  [{trend}]')
    print()
    print('  Session Flags:')
    for flag in summary['flags']:
        print(f'    {flag}')


if __name__ == '__main__':
    main()
