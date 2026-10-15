# Diffusion imputation for packet-loss audio

Research prototype for restoring missing audio packets with a conditional
score-based diffusion model derived from CSDI.

## Current pipeline

`src/exe_audio.py` performs the complete experimental pipeline:

1. loads matching clean audio, lossy audio, and packet-loss traces;
2. crops or pads audio to 10 seconds at 16 kHz;
3. converts it to a compressed complex STFT representation;
4. trains a conditional diffusion model on the missing STFT frames;
5. runs reverse diffusion, reconstructs WAV files, and evaluates them.

The current STFT setup is `n_fft=512`, `hop_length=256`, and
`win_length=512`. The network uses linear attention along the time and
frequency axes.

## Repository layout

- `src/exe_audio.py` — training and evaluation entry point.
- `src/dataset_packetloss.py` — dataset, masks, STFT, and train/valid/test splits.
- `src/main_model.py` — CSDI training loss and reverse diffusion.
- `src/diff_models.py` — diffusion network and attention blocks.
- `src/utils.py` — training loop, waveform reconstruction, and metrics.
- `src/diffusion_visual_test.py` — visualization of reverse diffusion steps.
- `config/base.yaml` — current experiment configuration.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The dataset is intentionally not stored in Git. Each item consists of:

```text
train/clean_signals/<item_id>.wav
train/lossy_signals/<item_id>.wav
train/traces/<item_id>_is_lost.txt
```

By default the entry point reads the three dataset directories from `train/`
next to the source tree. Different locations can be passed explicitly with
`--clean_dir`, `--lossy_dir`, and `--traces_dir`.

```bash
python src/exe_audio.py --device cuda:0 --config base.yaml
```

Evaluation can load either a checkpoint file or an experiment directory. New
experiments store the STFT normalization value in `config.json`, so inference
uses the same preprocessing as training.

```bash
python src/exe_audio.py \
  --device cuda:0 \
  --eval_only \
  --checkpoint save/audio_fold0_YYYYMMDD_HHMMSS/model_best.pth
```

To run an overfit diagnostic on selected items:

```bash
python src/exe_audio.py --device cuda:0 --item_id 140535,2106,209055
```

## Project status

The model can learn to reconstruct packet-loss regions in one-item and
small-subset overfit experiments. The pipeline now uses a held-out validation
loader by default, persists normalization metadata, records evaluation metrics
as JSON, and reports RMSE/MAE specifically inside packet-loss regions. The next
milestone is a controlled baseline experiment on a fixed train/valid/test
split.

Generated datasets, checkpoints, WAV files, and experiment folders are
excluded by `.gitignore`.

