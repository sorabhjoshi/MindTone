#!/usr/bin/env python3
"""Demo A — pick a random test-set clip and print its full mental-health
report. Requires the wav caches and dataframe splits already built (i.e.
run after train.py, or point --clip at any audio file directly).

Usage:
    python scripts/demo_single.py --clip path/to/some.wav
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from inference.mental_health import analyze_mental_health  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--clip', required=True, help='Path to an audio file')
    args = p.parse_args()

    print('=' * 72)
    print('  SINGLE UTTERANCE MENTAL HEALTH REPORT')
    print('=' * 72)
    report = analyze_mental_health(args.clip)
    print(f"  File       : {report['audio_file']}")
    print(f"  Predicted  : {report['emotion']['predicted']}  "
          f"({report['emotion']['confidence']:.1%} confidence)")
    print(f"  Uncertainty: {report['emotion']['uncertainty']:.3f}")
    print()
    print('  Emotion probabilities:')
    for em, prob in sorted(report['emotion']['probabilities'].items(), key=lambda x: -x[1]):
        bar = '#' * int(prob * 40)
        print(f'    {em:8s} {prob:5.3f} |{bar}')
    print()
    print('  Mental Health Indicators:')
    for cond, info in report['mental_health'].items():
        bar = '#' * int(info['score'] * 30)
        space = '.' * (30 - len(bar))
        print(f"    {cond:22s} {info['score']:5.3f} |{bar}{space}| {info['risk_level']}")
    print()
    print('  Flags:')
    for flag in report['flags']:
        print(f'    {flag}')


if __name__ == '__main__':
    main()
