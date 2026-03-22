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

- For **training** or **eval**, pass **`--data-root ...`** (default **`.`**); expected tree under that root: **`./data/SpeechCommands/`** (standard v0.02 layout).

For a **first-time download**, use **`--download`** on `train.py` and/or `eval.py` so torchaudio can fetch the dataset; otherwise place data under **`./data/SpeechCommands/`** and omit **`--download`**.

---

## Training

```bash
python train.py
# optional: download dataset if missing; custom root (still uses <data-root>/data/SpeechCommands/...)
python train.py --download --data-root ./data
python train.py --data-root /path/to/project_or_data_parent
```

`train.py` arguments:

- **`--data-root`** (default: **`./data`**) — passed as `root` to the dataset loader; layout stays **`./data/SpeechCommands/`**.
- **`--download`** (default: **off**) — set flag to let torchaudio fetch Speech Commands when needed.

Checkpoints and TensorBoard logs share the same run root:

- **Root:** `lightning_logs/speech_commands/`
- **Per training run:** Lightning creates a version folder `version_0`, `version_1`, … (incrementing each time you train from a clean or new run).
- **Checkpoints:** `lightning_logs/speech_commands/version_<N>/checkpoints/`
- **Best val-acc file:** `ModelCheckpoint` uses the pattern `best-val-acc-epoch…` (see `train.py`; exact suffix may include the epoch index Lightning adds).

Example (your machine may use a different `version_*`):

`lightning_logs/speech_commands/v1/checkpoints/best-val-acc-epochepoch=173.ckpt`

---

## TensorBoard

```bash
./launch_tensorboard.sh            # default port 7788
# or: ./launch_tensorboard.sh 6006
```

Opens logs from `lightning_logs/` (train/val metrics, LR, gradient norm, etc.).

---

## Evaluation (test set)

`eval.py` accepts **`--checkpoint`** / **`-c`**. If you omit it, the **default** checkpoint path is:

`./lightning_logs/speech_commands/v1/checkpoints/best-val-acc-epochepoch=173.ckpt`

(point this default at whatever `version_*/checkpoints/*.ckpt` you actually use after training).

```bash
# default checkpoint path (see above); default data root ./data
python eval.py

# override checkpoint and/or data location / download
python eval.py -c path/to/your.ckpt --data-root /path/to/project_or_data_parent
python eval.py --download   # fetch Speech Commands if missing under <data-root>/data/SpeechCommands/
```

`eval.py` also accepts **`--data-root`** (default **`./data`**) and **`--download`** (default **off**), same semantics as `train.py`.

PyTorch Lightning will print **test** metrics to the terminal.

---

## Reported test results

Results below are from one completed run (`trainer.test` on the test loader), matching the metrics table printed at the end of evaluation:

| Metric | Value |
|--------|--------|
| test/Loss | 0.1932 |
| test/acc | 0.9697 |
Full-precision values from that run:

- `test/Loss`: 0.19318167865276337  
- `test/acc`: 0.969650149345398

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
