"""Dataset over cached raw waveforms. No mel/CNN branch — see cache.py's
module docstring for why."""
import random

import torch
from torch.utils.data import Dataset

from data.audio import augment_wav


class SERDataset(Dataset):
    def __init__(self, wav_cache, labels, domains=None, augment=False, wav_lens=None):
        N = len(labels)
        assert len(wav_cache) == N
        self.wav_cache = wav_cache
        self.labels = labels
        self.domains = domains if domains is not None else [0] * N
        self.augment = augment
        # How many samples of each cached (fixed-length, zero-padded) wav
        # are real audio vs. padding. None means "treat as fully valid" —
        # used by eval paths that build a fresh SERDataset without a
        # matching wav_lens array.
        self.wav_lens = wav_lens if wav_lens is not None else [wav_cache.shape[1]] * N

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        wav = self.wav_cache[idx].clone().float()
        wav_len = int(self.wav_lens[idx])

        if self.augment and random.random() < 0.30:
            wav = augment_wav(wav, noise_prob=1.0, speed_prob=0.0)

        lbl = torch.tensor(self.labels[idx], dtype=torch.long)
        dom = torch.tensor(self.domains[idx], dtype=torch.long)
        wln = torch.tensor(wav_len, dtype=torch.long)
        return wav, wln, lbl, dom
