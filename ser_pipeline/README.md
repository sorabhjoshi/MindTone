# Speech Emotion Recognition Pipeline

Fine-tuned WavLM/wav2vec2 backbone for 6-class speech emotion recognition
(Angry, Disgust, Fear, Happy, Neutral, Sad), trained on RAVDESS + CREMA-D +
IEMOCAP, with a domain-adversarial head and a small mental-health scoring
layer built on top of the emotion predictions.

**Locked-in result: test WA = 71.46%, test UA = 70.63%** (TTA x3, EMA
weights, speaker-independent split). See "Experiment log" below before
changing the model config — every attempt to beat this regressed.

## Project layout

```
config.py              All constants — dataset paths, model/training hyperparameters
train.py               CLI: full training pipeline (data -> Stage1 -> train -> eval -> save)
infer.py               CLI: run a saved checkpoint on one or more audio files
data/
  loaders.py            RAVDESS / CREMA-D / IEMOCAP parsers -> unified DataFrame
  split.py              Speaker-independent train/val/test split + class weights + sampler
  audio.py              Wav loading, silence trimming, wav-level augmentation
  cache.py              Raw-waveform disk cache (not extracted features — backbone is fine-tuned live)
  dataset.py            SERDataset (PyTorch Dataset)
model/
  layers.py             GRL, layer-weighted sum, temporal attention pooling, focal loss, pairwise contrastive loss
  ser_model.py           SERModel — the full architecture
engine/
  ema.py                 Exponential moving average of model weights
  optim.py               Optimizer with layer-wise LR decay + LR scheduler
  amp.py                 Mixed-precision helpers (torch.amp / torch.cuda.amp compatibility)
  loop.py                train_epoch / evaluate / evaluate_tta
  stage1.py              Stage 1 backbone pretraining (emotion-adapt before full training)
  calibration.py         Post-hoc temperature + per-class vector scaling
inference/
  predictor.py           Standalone checkpoint -> prediction on one audio file
  mental_health.py        Heuristic mental-health indicator scoring
  session.py              Multi-utterance session aggregation (trend, mood stability)
scripts/
  demo_single.py          Single-clip mental-health report demo
  demo_session.py         Multi-clip session-summary demo
```

## Setup

```bash
pip install -r requirements.txt
```

Point the pipeline at your data (defaults assume Kaggle paths):

```bash
export SER_RAVDESS_ROOT=/path/to/ravdess
export SER_CREMAD_ROOT=/path/to/cremad
export SER_IEMOCAP_ROOT=/path/to/IEMOCAP_full_release
export SER_WORK_DIR=/path/to/output_dir     # caches, checkpoints, plots go here
```

## Training

```bash
python train.py
```

First run builds the raw-waveform cache (slow — real-time-ish per file for
resample + trim), then reuses it on subsequent runs. Options:

```bash
python train.py --skip-stage1     # skip backbone emotion-pretraining
python train.py --epochs 50       # override the epoch ceiling (early stopping still applies)
python train.py --no-plots        # skip matplotlib output (headless machines)
```

Output: `ser_model.pth` (self-contained checkpoint — architecture config +
weights + calibration + label names), `training_history.csv`,
`training_curves.png`, `confusion_matrix.png`.

## Inference

```bash
python infer.py clip.wav
python infer.py clip.wav --mental-health
python infer.py clip1.wav clip2.wav --json
```

Or from Python:

```python
from inference.predictor import predict_emotion
from inference.mental_health import analyze_mental_health

result = predict_emotion('clip.wav')
report = analyze_mental_health('clip.wav')
```

Session-level aggregation across multiple utterances:

```python
from inference.session import MentalHealthSession

session = MentalHealthSession(patient_id='p001')
for clip in ['a.wav', 'b.wav', 'c.wav']:
    session.add_utterance(clip)
summary = session.get_session_summary()
```

Demo scripts:

```bash
python scripts/demo_single.py --clip clip.wav
python scripts/demo_session.py a.wav b.wav c.wav
```

**Note on the mental-health layer:** the MH scores are hand-set weighted
combinations of emotion probabilities, not learned from labeled
mental-health data. Treat them as a coarse screening signal for a demo,
not a clinical or diagnostic tool.

## Architecture

Raw waveform -> fine-tuned WavLM/wav2vec2 backbone (last 4 transformer
layers unfrozen, rest frozen) -> layer-weighted sum of last 4 hidden
states -> temporal attention pooling -> projection -> classifier head.
A domain-adversarial head (gradient reversal layer) sits behind the
pooled representation, trained against the 3 source datasets — currently
disabled (`DOMAIN_LOSS_WEIGHT=0.0`, see experiment log). A pairwise
contrastive loss term adds extra penalty specifically on the
Fear<->Happy confusion pair.

Removed from the original notebook: a parallel mel-spectrogram + CNN
(EfficientNet-B0) branch with gated fusion. Confirmed dead code — the
final architecture (v13+) is wav2vec2/WavLM-only; the branch was still
being computed but never consumed by the model.

## Experiment log

Every one of these was tried on top of the 71.46%/70.63% baseline and
either landed flat or regressed. Recorded here so they aren't retried
blind:

| Change | Result | Verdict |
|---|---|---|
| `N_W2V_UNFROZEN` 4->6 + swap backbone to `superb/wav2vec2-base-superb-er` (IEMOCAP-pretrained) | test WA 67.42% | Worse — likely too specialized to IEMOCAP's 4-class scheme at Fear/Disgust's expense |
| `DOMAIN_LOSS_WEIGHT` 0.0->0.15 + added Happy<->Neutral contrastive term | test WA 70.33%, Happy recall dropped to 0.538 (target class got *worse*) | Worse — boosting a confused pair's loss doesn't guarantee it resolves the intended direction |
| `DOMAIN_LOSS_WEIGHT` 0.15 alone (isolated) | not confirmed to beat baseline | Neutral/inconclusive |
| Swap backbone to `wavlm-large` (24 layers, 1024-dim) | test accuracy 61% | Much worse — likely undertrained: only 4/24 layers unfrozen is a smaller fraction of the network than 4/12 on base, same epoch budget/LR schedule tuned for the smaller model |

**Decision:** stop tuning this architecture. If pursuing further gains,
lower-risk options not yet tried: multi-seed ensembling, stochastic
weight averaging (SWA) over late-training checkpoints, inspecting
misclassified Happy clips for a possible label-noise ceiling, or more
aggressive TTA — all inference/training-adjacent, none require touching
the model recipe above.
