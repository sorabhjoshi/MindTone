"""Raw-waveform loading, silence trimming, and wav-level augmentation.
Shared by cache-building, training, and inference so behavior is
guaranteed identical across all three."""
import random

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
import torchaudio.transforms as AT

import config

# Resample transforms are cached per (source_sr, target_sr) pair —
# rebuilding one from scratch every call is what made wav caching slow.
_resample_cache: dict = {}


def _get_resampler(orig_sr: int, target_sr: int) -> AT.Resample:
    key = (orig_sr, target_sr)
    if key not in _resample_cache:
        _resample_cache[key] = AT.Resample(orig_sr, target_sr)
    return _resample_cache[key]


def _read_audio_file(path: str):
    """Reads an audio file to a mono float32 torch tensor + its sample
    rate. Uses soundfile (libsndfile) rather than torchaudio.load()
    directly — torchaudio's own loading backend has changed across
    versions (recent releases require a separate torchcodec package for
    the same functionality soundfile already provides directly), so
    reading via soundfile avoids being coupled to whichever backend a
    given torchaudio release happens to ship with. Falls back to
    torchaudio only if soundfile can't read the format (e.g. some m4a/aac
    files, which libsndfile doesn't support)."""
    try:
        data, sr = sf.read(path, dtype='float32', always_2d=True)  # (T, channels)
        wav = torch.from_numpy(data.T)  # (channels, T)
    except Exception:
        import torchaudio
        wav, sr = torchaudio.load(path)
    return wav, sr


def trim_silence(wav: torch.Tensor, sr: int,
                  thresh_ratio: float = config.SIL_THRESH_RATIO,
                  frame_ms: int = config.SIL_FRAME_MS,
                  pad_ms: int = config.SIL_PAD_MS) -> torch.Tensor:
    """Energy-based leading/trailing silence trim, relative to the clip's
    own peak loudness (not an absolute threshold) so it adapts per file."""
    frame_len = max(1, int(sr * frame_ms / 1000))
    n_frames = wav.shape[0] // frame_len
    if n_frames < 2:
        return wav
    frames = wav[:n_frames * frame_len].reshape(n_frames, frame_len)
    energy = frames.pow(2).mean(dim=1).sqrt()
    peak = energy.max()
    if peak < 1e-6:
        return wav  # effectively silent file
    active = energy > (peak * thresh_ratio)
    if not active.any():
        return wav
    idx = active.nonzero().squeeze(-1)
    first, last = idx[0].item(), idx[-1].item()
    pad_frames = max(1, int(pad_ms / frame_ms))
    start = max(0, (first - pad_frames) * frame_len)
    end = min(wav.shape[0], (last + 1 + pad_frames) * frame_len)
    return wav[start:end]


def load_wav(path: str, target_sr: int = config.SAMPLE_RATE,
             max_sec: float = config.MAX_SECONDS):
    """Returns (wav, valid_len). valid_len is how many samples of the
    fixed-length output are real audio vs. zero-padding, used to build an
    attention mask so short clips aren't treated as if padding were signal."""
    wav, sr = _read_audio_file(path)
    if sr != target_sr:
        wav = _get_resampler(sr, target_sr)(wav)
    if wav.shape[0] > 1:
        wav = wav.mean(0, keepdim=True)
    wav = wav.squeeze(0)

    wav = trim_silence(wav, target_sr)

    target_len = int(max_sec * target_sr)
    T = wav.shape[0]
    if T < target_len:
        valid_len = T
        wav = F.pad(wav, (0, target_len - T))
    else:
        valid_len = target_len
        start = (T - target_len) // 2
        wav = wav[start: start + target_len]
    peak = wav.abs().max()
    if peak > 1e-6:
        wav = wav / peak
    return wav, valid_len


def augment_wav(wav: torch.Tensor, noise_prob: float = config.WAV_NOISE_PROB,
                 speed_prob: float = config.WAV_SPEED_PROB) -> torch.Tensor:
    """Random wav-level augmentation: Gaussian noise at a random SNR, and/or
    speed perturbation via resampling (changes pitch + duration, then
    re-pad/crop back to a fixed length)."""
    if random.random() < noise_prob:
        snr_db = random.uniform(*config.WAV_NOISE_SNR)
        signal_power = wav.pow(2).mean().clamp(min=1e-12)
        noise_power = signal_power / (10 ** (snr_db / 10))
        wav = (wav + torch.randn_like(wav) * noise_power.sqrt()).clamp(-1, 1)

    if random.random() < speed_prob:
        factor = random.uniform(*config.WAV_SPEED_RANGE)
        new_sr = int(config.SAMPLE_RATE * factor)
        wav_2d = wav.unsqueeze(0)
        wav_speed = _get_resampler(config.SAMPLE_RATE, new_sr)(wav_2d).squeeze(0)
        target_len = int(config.MAX_SECONDS * config.SAMPLE_RATE)
        T2 = wav_speed.shape[0]
        if T2 < target_len:
            wav_speed = F.pad(wav_speed, (0, target_len - T2))
        else:
            start = (T2 - target_len) // 2
            wav_speed = wav_speed[start: start + target_len]
        peak = wav_speed.abs().max()
        if peak > 1e-6:
            wav_speed = wav_speed / peak
        wav = wav_speed

    return wav
