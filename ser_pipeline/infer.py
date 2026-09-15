#!/usr/bin/env python3
"""
Run trained-model inference on one or more audio files.

Usage:
    python infer.py path/to/clip.wav
    python infer.py clip1.wav clip2.wav clip3.wav
    python infer.py clip.wav --mental-health          # also print MH indicators
    python infer.py clip.wav --model-path other.pth
"""
import argparse
import json

import config
from inference.mental_health import analyze_mental_health
from inference.predictor import predict_emotion


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('audio_paths', nargs='+', help='One or more audio file paths')
    p.add_argument('--model-path', default=None,
                    help=f'Checkpoint path (default: {config.MODEL_PATH})')
    p.add_argument('--mental-health', action='store_true',
                    help='Also compute heuristic mental-health indicators')
    p.add_argument('--json', action='store_true', help='Print raw JSON instead of a formatted report')
    return p.parse_args()


def print_emotion_report(result: dict):
    print(f"  File       : {result['audio_file']}")
    print(f"  Predicted  : {result['predicted_emotion']}  ({result['confidence']:.1%} confidence)")
    print(f"  Uncertainty: {result['uncertainty']:.3f}")
    print('  Probabilities:')
    for em, p in sorted(result['class_probabilities'].items(), key=lambda x: -x[1]):
        bar = '#' * int(p * 40)
        print(f'    {em:8s} {p:5.3f} |{bar}')


def print_mh_report(report: dict):
    print()
    print('  Mental Health Indicators:')
    for cond, info in report['mental_health'].items():
        bar = '#' * int(info['score'] * 30)
        space = '.' * (30 - len(bar))
        print(f"    {cond:22s} {info['score']:5.3f} |{bar}{space}| {info['risk_level']}")
    print('  Flags:')
    for flag in report['flags']:
        print(f'    {flag}')


def main():
    args = parse_args()
    for path in args.audio_paths:
        print('=' * 65)
        if args.mental_health:
            report = analyze_mental_health(path)
            if args.json:
                print(json.dumps(report, indent=2))
            else:
                print_emotion_report({
                    'audio_file': report['audio_file'],
                    'predicted_emotion': report['emotion']['predicted'],
                    'confidence': report['emotion']['confidence'],
                    'uncertainty': report['emotion']['uncertainty'],
                    'class_probabilities': report['emotion']['probabilities'],
                })
                print_mh_report(report)
        else:
            result = predict_emotion(path, model_path=args.model_path)
            if args.json:
                print(json.dumps(result, indent=2))
            else:
                print_emotion_report(result)
    print('=' * 65)


if __name__ == '__main__':
    main()
