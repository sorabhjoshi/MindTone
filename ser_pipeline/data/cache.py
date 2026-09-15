"""Raw-waveform disk cache. wav2vec2/WavLM is fine-tuned live inside the
model on raw audio, so what's cached here is preprocessed waveforms, not
extracted features — caching avoids re-decoding/resampling/trimming every
file on every epoch.

Note: this pipeline is wav2vec2/WavLM-only (see model/ser_model.py) — an
earlier version also had a parallel mel-spectrogram + CNN branch, but that
branch was found to be entirely unused by the final architecture and was
removed here rather than carried forward as dead code.
"""
import os

import pandas as pd
import torch
from tqdm.auto import tqdm

from data.audio import load_wav


def build_wav_cache(df_split: pd.DataFrame, cache_path: str, split_name: str = 'split'):
    if os.path.exists(cache_path):
        d = torch.load(cache_path, map_location='cpu', weights_only=True)
        wavs, lens = d['wav'], d['len']
        print(f'[{split_name}] Wav cache loaded: {tuple(wavs.shape)}')
        assert wavs.shape[0] == len(df_split), (
            f'Size mismatch! Delete {cache_path} and rerun.')
        return wavs, lens

    paths = df_split['path'].tolist()
    print(f'[{split_name}] Loading {len(paths)} raw waveforms ...')
    results = [load_wav(p) for p in tqdm(paths, desc=f'wav [{split_name}]')]
    wavs = torch.stack([r[0] for r in results])
    lens = torch.tensor([r[1] for r in results], dtype=torch.long)
    torch.save({'wav': wavs, 'len': lens}, cache_path)
    print(f'[{split_name}] Saved: {tuple(wavs.shape)} -> {cache_path}')
    return wavs, lens
