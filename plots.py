# --- load created models to test on fd00x test files ---

import numpy as np
from keras.models import load_model
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.pyplot as plt

from pathlib import Path
import joblib
from keras.models import load_model as keras_load_model
import random
from typing import Iterable, Dict, List, Optional, Tuple

def load_any_model(path):
    """
    Load a model given its file path, handling both Keras (.keras/.h5) and joblib (.joblib/.pkl).
    """
    path = Path(path)
    suf = path.suffix.lower()
    if suf in {".keras", ".h5", ".hdf5"}:
        return keras_load_model(path)
    if suf in {".joblib", ".pkl"}:
        return joblib.load(path)
    raise ValueError(f"Unrecognized model suffix for {path.name}")

def load_models(model_paths: dict):
    """
    model_paths: dict like {"Base": "...joblib", "LSTM": "...keras", "CNN": "...keras", "CNN-LSTM": "...keras"}
    returns: dict {name: model_object}
    """
    models = {}
    for name, path in model_paths.items():
        models[name] = load_any_model(path)
    return models

def build_model_paths(dataset=None, seq_len=None, strategy="last", art_dir=None,
                      include=("Base", "LSTM", "CNN", "CNN-LSTM")):
    """
    Build default file paths for your saved models in <ART_DIR>/models/…
    Base:      base_linear_{dataset.lower()}_seq{seq_len}_{strategy}.joblib
    LSTM:      lstm_{dataset.lower()}_seq{seq_len}.keras
    CNN:       cnn_{dataset.lower()}_seq{seq_len}.keras
    CNN-LSTM:  cnn_lstm_{dataset.lower()}_seq{seq_len}.keras
    """
    if dataset is None:
        dataset = globals().get("DATASET", "FD001")
    if seq_len is None:
        seq_len = globals().get("SEQ_LEN", 30)
    if art_dir is None:
        art_dir = globals().get("ART_DIR", Path.cwd() / f"{dataset} data & artefacts")

    models_dir = Path(art_dir) / "models"
    ds = dataset.lower()

    paths = {}
    if "Base" in include:
        paths["Base"] = str(models_dir / f"base_linear_{ds}_seq{seq_len}_{strategy}.joblib")
    if "LSTM" in include:
        paths["LSTM"] = str(models_dir / f"lstm_{ds}_seq{seq_len}.keras")
    if "CNN" in include:
        paths["CNN"] = str(models_dir / f"cnn_{ds}_seq{seq_len}.keras")
    if "CNN-LSTM" in include:
        paths["CNN-LSTM"] = str(models_dir / f"cnn_lstm_{ds}_seq{seq_len}.keras")
    return paths


def plot_true_vs_pred(y_true, y_pred, model_name="Model", savepath=None):
    """
    Scatter plot: True RUL vs Predicted RUL for a single model.
    
    Args:
        y_true (array-like): Ground truth RUL values.
        y_pred (array-like): Predicted RUL values.
        model_name (str): Name of the model for labeling.
        savepath (str or None): If given, save the figure to this path.
    """
    plt.figure(figsize=(6, 6))
    plt.scatter(y_true, y_pred, alpha=0.6, edgecolor="k")
    max_val = max(y_true.max(), y_pred.max())
    plt.plot([0, max_val], [0, max_val], "r--", lw=2, label="Ideal")
    plt.xlabel("True RUL")
    plt.ylabel("Predicted RUL")
    plt.title(f"True vs Predicted RUL ({model_name})")
    plt.legend()
    plt.grid(True)
    if savepath:
        plt.savefig(savepath, dpi=300, bbox_inches="tight")
    plt.show()



def plot_residuals(y_true, y_pred, model_name="Model", kind="box", savepath=None):
    """
    Plot residuals (Predicted - True) for a single model.
    
    Args:
        y_true (array-like): Ground truth RUL values.
        y_pred (array-like): Predicted RUL values.
        model_name (str): Label for the model.
        kind (str): "box" or "violin" for plot type.
        savepath (str or None): If given, save the figure.
    """
    errors = y_pred - y_true
    plt.figure(figsize=(6, 4))
    
    if kind == "violin":
        sns.violinplot(y=errors)
    else:
        sns.boxplot(y=errors)
    
    plt.axhline(0, color="r", linestyle="--", lw=2, label="Zero Error")
    plt.ylabel("Residual (Predicted - True)")
    plt.title(f"Residual Distribution ({model_name})")
    plt.legend()
    plt.grid(True, axis="y")
    if savepath:
        plt.savefig(savepath, dpi=300, bbox_inches="tight")
    plt.show()

def plot_per_engine_bars(y_true, y_pred, unit_ids, model_name="Model", n_samples=30, savepath=None, seed=42):
    """
    Bar plot comparing Actual vs Predicted RUL for a random sample of engines.
    
    Args:
        y_true (array-like): Ground truth RUL values (aligned with unit_ids).
        y_pred (array-like): Predicted RUL values (aligned with unit_ids).
        unit_ids (array-like): Engine/unit identifiers for each sample.
        model_name (str): Name of the model for labeling.
        n_samples (int): Number of engines to randomly sample.
        savepath (str or None): If given, save the figure.
        seed (int): Random seed for reproducibility.
    """
    random.seed(seed)
    unique_ids = np.unique(unit_ids)
    chosen_ids = random.sample(list(unique_ids), min(n_samples, len(unique_ids)))

    # Collect true & pred for chosen engines
    true_sample, pred_sample, labels = [], [], []
    for uid in chosen_ids:
        mask = unit_ids == uid
        # last entry corresponds to the test RUL label
        true_sample.append(y_true[mask][-1])
        pred_sample.append(y_pred[mask][-1])
        labels.append(str(uid))

    x = np.arange(len(chosen_ids))
    width = 0.35

    plt.figure(figsize=(12, 6))
    plt.bar(x - width/2, true_sample, width, label="Actual")
    plt.bar(x + width/2, pred_sample, width, label="Predicted")
    plt.xticks(x, labels, rotation=45)
    plt.xlabel("Engine ID (sampled)")
    plt.ylabel("RUL")
    plt.title(f"Actual vs Predicted RUL (Sampled Engines) - {model_name}")
    plt.legend()
    plt.tight_layout()
    if savepath:
        plt.savefig(savepath, dpi=300, bbox_inches="tight")
    plt.show()
    
    
def plot_metric_comparison(metrics_list,
                           dataset_name: str = "FD001",
                           savepath: Optional[str] = None):
    """
    Grouped bar chart of RMSE/MAE across models.
    Accepts either:
      - list of dicts: [{"model":"LSTM","RMSE":15.2,"MAE":11.3}, ...]
      - DataFrame with columns: model/Model, RMSE, MAE
    """
    # Build DataFrame
    df = metrics_list.copy() if isinstance(metrics_list, pd.DataFrame) else pd.DataFrame(metrics_list)
    if df.empty:
        print("No metrics to plot.")
        return

    # Normalise column names
    if "model" not in df.columns and "Model" in df.columns:
        df = df.rename(columns={"Model": "model"})

    # Validate required columns
    required = {"model", "RMSE", "MAE"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns for plotting: {missing}")

    # Keep only what we need, coerce to numeric
    df = df[["model", "RMSE", "MAE"]].copy()
    df[["RMSE", "MAE"]] = df[["RMSE", "MAE"]].astype(float)
    df = df.set_index("model")

    # Plot
    ax = df.plot(kind="bar", figsize=(10, 6))
    ax.set_title(f"RMSE / MAE Comparison — {dataset_name}")
    ax.set_ylabel("Error")
    ax.set_xlabel("")
    ax.grid(True, axis="y", alpha=0.3)
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.show()
