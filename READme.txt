Remaining Useful Life (RUL) Prediction on NASA C-MAPSS

LSTM vs CNN (with CNN-LSTM hybrid) using PyTorch-Keras

Predicting Remaining Useful Life (RUL) for turbofan engines with deep learning, evaluated on the NASA C-MAPSS FD00x datasets. This repo provides a reproducible, modular pipeline: data loading → preprocessing → model training/evaluation → visualisation and comparisons.

Why this repo?
Compare sequence models (LSTM, CNN, CNN-LSTM) against a simple linear baseline to understand accuracy/robustness across FD001–FD004.

Features

Data utilities for C-MAPSS (train/test + RUL files, proper column names) and quick inspection.

Pre-processing: per-condition standardisation, capped RUL labels, sliding windows, unit-wise splits, and NPZ artefact saving/loading.

Models:

Baseline linear regression on engineered features (last/mean/flat).

LSTM (stacked 64→32 + dropout).

CNN (two 1-D conv blocks → GAP → Dense).

CNN-LSTM hybrid (Conv blocks → LSTM → Dense).

Evaluation: RMSE & MAE helpers + tidy printouts.

Visualisation: parity plots, residuals, per-engine bars, and grouped metric comparisons; model loader for saved artefacts.

Repository structure
.
├─ data/                     # Place raw C-MAPSS files here (train_FD001.txt, test_FD001.txt, RUL_FD001.txt, …)
├─ artefacts/                # Saved NPZ, models, figures (created by you during runs)
│  └─ models/                # *.keras / *.h5 / *.joblib
├─ notebooks/                #  Jupyter notebooks (optional)
├─ cnn_model.py              # 1-D temporal CNN model & pipeline :
├─ lstm_model.py             # LSTM model & pipeline 
├─ cnn_lstm_model.py         # CNN→LSTM hybrid & pipeline 
├─ base_model.py             # Baseline linear regression on features 
├─ data_loader.py            # C-MAPSS loaders & quick inspection :
├─ pre_processing.py         # RUL labels, scaling per condition, windows, splits
├─ evaluator.py              # RMSE/MAE and evaluation wrapper :
└─ plots.py                  # All figures + model loaders and path builders

 Data

Source: NASA C-MAPSS (FD001–FD004).

Files expected (per dataset):
train_FD00x.txt, test_FD00x.txt, RUL_FD00x.txt under data/.

data_loader.py sets correct column names: unit_number, time_in_cycles, op_setting_1..3, sensor_measurement_1..21.

Environment

Python ≥ 3.10 recommended

Keras 3 with PyTorch backend (no TensorFlow)

Typical install:

pip install torch torchvision torchaudio
pip install keras numpy pandas scikit-learn matplotlib seaborn joblib


All model files defensively set:

os.environ.setdefault("KERAS_BACKEND", "torch")

so Keras uses the PyTorch backend.

Mac notes (Apple Silicon)
If you hit MPS quirks, you can force CPU in notebooks by setting PyTorch to CPU and/or using PYTORCH_ENABLE_MPS_FALLBACK=1.



 Acknowledgements

NASA C-MAPSS turbofan engine datasets.

Keras 3 running on the PyTorch backend (no TensorFlow).

Project structure inspired by standard PHM/RUL workflows.

Zero-Clause BSD (0BSD) License

Copyright (C) 2025 Mamadou Talibe Bah

Permission to use, copy, modify, and/or distribute this software for any purpose
with or without fee is hereby granted.

THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES WITH
REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY AND
FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY SPECIAL, DIRECT,
INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES WHATSOEVER RESULTING FROM LOSS
OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER
TORTIOUS ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE OF
THIS SOFTWARE.
