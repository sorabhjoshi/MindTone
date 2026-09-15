"""Speaker-independent train/val/test split, plus the class-imbalance
correction (WeightedRandomSampler) built from the resulting train split."""
import numpy as np
import pandas as pd
import torch
from torch.utils.data import WeightedRandomSampler

import config


def speaker_split(df: pd.DataFrame, val_frac: float = 0.125,
                   test_frac: float = 0.125, seed: int = config.SEED):
    train_idx, val_idx, test_idx = [], [], []
    for ds in df['dataset'].unique():
        sub = df[df['dataset'] == ds]
        speakers = sorted(sub['speaker'].unique())
        n = len(speakers)

        if n < 4:
            # Guard for a corpus with too few speakers to hold any out.
            train_idx.extend(sub.index.tolist())
            print(f'  [{ds}] only {n} speakers -> ALL {len(sub)} samples go to TRAIN')
            continue

        rng = np.random.RandomState(seed)
        rng.shuffle(speakers)
        n_test = max(1, int(n * test_frac))
        n_val = max(1, int(n * val_frac))
        test_spk = set(speakers[:n_test])
        val_spk = set(speakers[n_test:n_test + n_val])
        train_spk = set(speakers[n_test + n_val:])
        train_idx.extend(sub[sub['speaker'].isin(train_spk)].index.tolist())
        val_idx.extend(sub[sub['speaker'].isin(val_spk)].index.tolist())
        test_idx.extend(sub[sub['speaker'].isin(test_spk)].index.tolist())
    return train_idx, val_idx, test_idx


def make_splits(df: pd.DataFrame):
    """Returns (df_train, df_val, df_test), each with a 'domain' column
    added, plus the class_weights tensor and WeightedRandomSampler for
    the train split."""
    train_idx, val_idx, test_idx = speaker_split(df)
    df_train = df.loc[train_idx].reset_index(drop=True)
    df_val = df.loc[val_idx].reset_index(drop=True)
    df_test = df.loc[test_idx].reset_index(drop=True)

    train_spk, val_spk, test_spk = (set(df_train['speaker']),
                                     set(df_val['speaker']),
                                     set(df_test['speaker']))
    assert not (train_spk & val_spk), 'OVERLAP: train n val'
    assert not (train_spk & test_spk), 'OVERLAP: train n test'
    assert not (val_spk & test_spk), 'OVERLAP: val n test'

    print('Speaker-independent split:')
    print(f'  Train : {len(df_train):5d} files  |  {df_train["speaker"].nunique():3d} speakers')
    print(f'  Val   : {len(df_val):5d} files  |  {df_val["speaker"].nunique():3d} speakers')
    print(f'  Test  : {len(df_test):5d} files  |  {df_test["speaker"].nunique():3d} speakers')
    print('  No speaker overlap confirmed.')

    for split in (df_train, df_val, df_test):
        split['domain'] = split['dataset'].map(config.DS2I)

    counts = np.array([(df_train['emotion'] == i).sum()
                        for i in range(config.NUM_CLASSES)], dtype=float)
    class_weights = torch.tensor(
        counts.sum() / (config.NUM_CLASSES * counts), dtype=torch.float)
    print(f'\nClass weights: {class_weights.numpy().round(3)}')

    sample_weights = torch.tensor(
        [class_weights[e].item() for e in df_train['emotion'].tolist()],
        dtype=torch.float)
    sampler = WeightedRandomSampler(
        sample_weights, num_samples=len(sample_weights), replacement=True)
    print(f'WeightedRandomSampler ready ({len(sample_weights)} train samples)')

    return df_train, df_val, df_test, class_weights, sampler
