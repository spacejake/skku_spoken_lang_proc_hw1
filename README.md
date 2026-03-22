# Spoken Language Processing — HW1: Lightweight Keyword Spotting (KWS)

## Assignment overview

Train a **lightweight keyword spotting** model on the **Google Speech Commands Dataset v2 (GSC v2)**. Background reading: *Streaming Keyword Spotting on Mobile Devices* (Google, 2020).

### Task and data

- **Audio:** 16 kHz sampling rate, 1 s clips.
- **Classes (12):** ten commands — `yes`, `no`, `up`, `down`, `left`, `right`, `on`, `off`, `stop`, `go` — plus **`silence`** and **`unknown`**.
- **Split (approx. 8 : 1 : 1):** train **36,923** · validation **4,445** · test **4,890**.  
    - **Actual Split:** train **84,843** · validation **9,981** · test **11,005**.
- **Input features:** **log-mel spectrogram** with a **40 ms** window, **20 ms** hop (overlap), and **80** mel bins (see `model/encoder.py`).

### Model and training requirements

- **Model size:** total parameters **≤ 2.5M** (this repo’s encoder is sized for that budget).
- **Architecture:** any reasonable choice (here: ConvNeXt-style encoder + classifier); streaming not required.
- **Augmentation:** at least one input augmentation (here: **time masking** on training batches via `dataset/augmentations.py`).
- **Regularization:** at least two techniques (here e.g. **dropout** on the classification head, **AdamW weight decay**, plus optional structural regularities in the stack).
- **Logging:** track **train loss**, **train/validation accuracy**, **learning rate**, and **gradient norm** in **TensorBoard** (PyTorch Lightning + `TensorBoardLogger`).
- **Evaluation:** report performance on the **held-out test** set.

### Report (course policy)

- PDF, **≤ 3 pages**, **11 pt** body text; include **experiment tracking** (figures or screenshots from TensorBoard).

---

## Environment (recommended: [uv](https://github.com/astral-sh/uv))

From the project root (`hw1/`):

```bash
uv venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
uv pip install -r requirements.txt
```

**System dependency:** `ffmpeg` is listed in `requirements.txt` for some audio tooling; on macOS you can install it with Homebrew (`brew install ffmpeg`) if imports fail.

Run scripts with the venv active, or prefix with `uv run` (e.g. `uv run python train.py`).

---

## Data layout

This codebase follows the **torchaudio** Speech Commands layout under the project root:

- Set `data_root = './'` in `train.py` / `eval.py` (default).
- Expected tree: `./data/SpeechCommands/` with the standard v0.02 folder structure.

For a **first-time download**, set `download_option = True` in `train.py` (and `eval.py` if needed) so torchaudio can fetch the dataset; otherwise place an existing copy under `./data/SpeechCommands/` and keep `download_option = False`.

---

## Training

```bash
python train.py
```

Checkpoints and TensorBoard runs are written under `lightning_logs/` (run name `speech_commands`). The **best validation accuracy** checkpoint is saved by `ModelCheckpoint`.

---

## TensorBoard

```bash
./launch_tensorboard.sh            # default port 7788
# or: ./launch_tensorboard.sh 6006
```

Opens logs from `lightning_logs/` (train/val metrics, LR, gradient norm, etc.).

---

## Evaluation (test set)

1. In `eval.py`, set `ConvNextASR.load_from_checkpoint(...)` to your **saved checkpoint** path (under `lightning_logs/speech_commands/version_*/checkpoints/`; filename pattern `best-val-acc-epoch*.ckpt`).
2. Run:

```bash
python eval.py
```

PyTorch Lightning will print **test** metrics to the terminal.

---

## Reported test results

Results below are from one completed run (`trainer.test` on the test loader), matching the metrics table printed at the end of evaluation:

| Metric | Value |
|--------|--------|
| test/Loss | 0.1932 |
| test/acc | 0.9697 |
| test/macro_f1 | 0.9527 |
| test/micro_f1 | 0.9697 |
| test/weighted_f1 | 0.9696 |

Full-precision values from that run:

- `test/Loss`: 0.19318167865276337  
- `test/acc`: 0.969650149345398  
- `test/macro_f1`: 0.952729344367981  
- `test/micro_f1`: 0.969650149345398  
- `test/weighted_f1`: 0.969610869884491  

---

## Project map

| Path | Role |
|------|------|
| `train.py` | Training loop, data loaders, Lightning `Trainer.fit` |
| `eval.py` | Load checkpoint and `Trainer.test` |
| `dataset/speechcommands.py` | 12-class GSC split and labels |
| `dataset/augmentations.py` | Training-time augmentation |
| `model/speechcommand.py` | Lightning module (`ConvNextASR`) |
| `model/encoder.py` | Log-mel front-end and encoder |
| `training/tensorboard_utils.py` | Extra TensorBoard helpers |
| `launch_tensorboard.sh` | Local TensorBoard server |
