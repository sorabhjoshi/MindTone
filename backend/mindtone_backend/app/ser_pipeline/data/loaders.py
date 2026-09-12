"""Parses the three raw corpora into a single unified DataFrame of
{path, label, emotion, speaker, dataset} records."""
import glob
import os

import pandas as pd

import config

RAVDESS_MAP = {
    '01': 'Neutral', '03': 'Happy', '04': 'Sad',
    '05': 'Angry', '06': 'Fear', '07': 'Disgust',
}

CREMAD_MAP = {'ANG': 'Angry', 'DIS': 'Disgust', 'FEA': 'Fear',
              'HAP': 'Happy', 'NEU': 'Neutral', 'SAD': 'Sad'}

# 'fru' (Frustration) and 'exc' (Excited) are dropped rather than merged
# into Angry/Happy — folding them in would quietly change what those
# labels mean in the combined dataset. 'sur'/'oth' aren't in our label
# set at all, and 'xxx' means annotators didn't agree on any label.
IEMOCAP_MAP = {'ang': 'Angry', 'dis': 'Disgust', 'fea': 'Fear',
               'hap': 'Happy', 'neu': 'Neutral', 'sad': 'Sad'}


def parse_ravdess(root: str) -> list[dict]:
    records = []
    for path in sorted(glob.glob(os.path.join(root, '**', '*.wav'), recursive=True)):
        fname = os.path.basename(path)
        parts = fname.replace('.wav', '').split('-')
        if len(parts) != 7:
            continue
        modality, _, emotion_code, _, _, _, actor = parts
        if modality != '03':
            continue
        label = RAVDESS_MAP.get(emotion_code)
        if label is None:
            continue
        records.append({'path': path, 'label': label, 'emotion': config.L2I[label],
                         'speaker': f'RAV_{int(actor):02d}', 'dataset': 'RAVDESS'})
    return records


def parse_cremad(root: str) -> list[dict]:
    records = []
    for path in sorted(glob.glob(os.path.join(root, '**', '*.wav'), recursive=True)):
        fname = os.path.basename(path)
        parts = fname.replace('.wav', '').split('_')
        if len(parts) < 3:
            continue
        actor_id, emotion_code = parts[0], parts[2]
        label = CREMAD_MAP.get(emotion_code)
        if label is None:
            continue
        records.append({'path': path, 'label': label, 'emotion': config.L2I[label],
                         'speaker': f'CRE_{actor_id}', 'dataset': 'CREMA-D'})
    return records


def parse_iemocap(root: str) -> list[dict]:
    """IEMOCAP has no per-file label in the filename — labels live in
    per-dialog EmoEvaluation/*.txt annotation files, one line per
    conversational turn. Each session is a fixed pair of actors (1 male,
    1 female); the turn name's trailing letter (F/M) tells you which of
    that session's 2 actors spoke."""
    records = []
    for sess_dir in sorted(glob.glob(os.path.join(root, 'Session*'))):
        session_num = os.path.basename(sess_dir).replace('Session', '')
        emo_files = sorted(glob.glob(
            os.path.join(sess_dir, 'dialog', 'EmoEvaluation', '*.txt')))
        for emo_path in emo_files:
            dialog_name = os.path.basename(emo_path).replace('.txt', '')
            with open(emo_path, 'r', errors='ignore') as f:
                for line in f:
                    if not line.startswith('['):
                        continue
                    parts = line.strip().split('\t')
                    if len(parts) < 3:
                        continue
                    turn_name, code = parts[1], parts[2]
                    label = IEMOCAP_MAP.get(code)
                    if label is None:
                        continue
                    gender = turn_name[-4] if len(turn_name) >= 4 else ''
                    if gender not in ('F', 'M'):
                        continue
                    wav_path = os.path.join(sess_dir, 'sentences', 'wav',
                                             dialog_name, turn_name + '.wav')
                    if not os.path.exists(wav_path):
                        continue
                    records.append({'path': wav_path, 'label': label,
                                     'emotion': config.L2I[label],
                                     'speaker': f'IEM_{session_num}{gender}',
                                     'dataset': 'IEMOCAP'})
    return records


def load_all_records(ravdess_root: str = None, cremad_root: str = None,
                      iemocap_root: str = None) -> pd.DataFrame:
    ravdess_root = ravdess_root or config.RAVDESS_ROOT
    cremad_root = cremad_root or config.CREMAD_ROOT
    iemocap_root = iemocap_root or config.IEMOCAP_ROOT

    records = parse_ravdess(ravdess_root)
    records += parse_cremad(cremad_root)
    records += parse_iemocap(iemocap_root)
    df = pd.DataFrame(records).reset_index(drop=True)

    print(f'Total files : {len(df)}')
    print(f'Speakers    : {df["speaker"].nunique()}')
    print('\nPer dataset:')
    for ds in config.DATASET_NAMES:
        sub = df[df['dataset'] == ds]
        print(f'  {ds:8s} -> {len(sub):5d} files  |  {sub["speaker"].nunique():3d} speakers')
    print('\nClass distribution:')
    for i, name in enumerate(config.LABEL_NAMES):
        count = (df['emotion'] == i).sum()
        print(f'  [{i}] {name:8s} : {count:5d}')
    return df
