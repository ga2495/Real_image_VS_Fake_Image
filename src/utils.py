"""
utils.py
========
Shared utilities: model compilation, callbacks, evaluation, and visualisation.
"""

import logging
import os

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import keras
from tensorflow.keras import callbacks

from configs.config import LABEL_SMOOTHING, LEARNING_RATE, IMG_SIZE, CLASSIFICATION_THRESHOLD

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Mixed precision
# ---------------------------------------------------------------------------

def set_mixed_precision(policy: str = "mixed_float16") -> None:
    """Enable Keras mixed-precision training."""
    tf.keras.mixed_precision.set_global_policy(policy)
    p = tf.keras.mixed_precision.global_policy()
    logger.info(
        "Mixed precision enabled – compute: %s, variable: %s",
        p.compute_dtype,
        p.variable_dtype,
    )


# ---------------------------------------------------------------------------
# Model compilation helpers
# ---------------------------------------------------------------------------

def compile_model_single(model: tf.keras.Model, lr: float = LEARNING_RATE) -> None:
    """Compile model for single-output training (Stage 1)."""
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss=keras.losses.BinaryCrossentropy(label_smoothing=LABEL_SMOOTHING),
        metrics=[
            "accuracy",
            keras.metrics.Precision(name="precision"),
            keras.metrics.Recall(name="recall"),
            keras.metrics.AUC(name="auc"),
        ],
    )


def compile_model_multi(model: tf.keras.Model, lr: float = LEARNING_RATE) -> None:
    """Compile model for multi-output training (Stages 2 & 3)."""
    loss_fn = keras.losses.BinaryCrossentropy(label_smoothing=LABEL_SMOOTHING)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss={
            "main_output": loss_fn,
            "color_output": loss_fn,
            "freq_output": loss_fn,
        },
        loss_weights={
            "main_output": 1.0,
            "color_output": 0.3,
            "freq_output": 0.3,
        },
        metrics={
            "main_output": ["accuracy", keras.metrics.AUC(name="auc")],
            "color_output": ["accuracy"],
            "freq_output": ["accuracy"],
        },
    )


# ---------------------------------------------------------------------------
# Training callbacks
# ---------------------------------------------------------------------------

def get_callbacks(monitor: str = "val_loss") -> list:
    """
    Return standard Keras callbacks.

    Includes:
      - EarlyStopping  (patience=6, restores best weights)
      - ReduceLROnPlateau (factor=0.5, patience=3)
    """
    return [
        callbacks.EarlyStopping(
            monitor=monitor,
            patience=6,
            restore_best_weights=True,
            verbose=1,
        ),
        callbacks.ReduceLROnPlateau(
            monitor=monitor,
            factor=0.5,
            patience=3,
            min_lr=1e-7,
            verbose=1,
        ),
    ]


# ---------------------------------------------------------------------------
# Model persistence & Resolution
# ---------------------------------------------------------------------------

def get_custom_objects() -> dict:
    """Return dictionary of custom layers and functions for FDCS-Net V4 deserialization."""
    from model import (
        preprocess_fn,
        l2_norm_fn,
        l2_norm_output_shape,
        stack_fn,
        expand_dims_fn,
        cast_fn,
        weighted_sum_fn,
        compute_color_stability_map,
        compute_fft_magnitude_map,
    )
    return {
        "preprocess_fn": preprocess_fn,
        "l2_norm_fn": l2_norm_fn,
        "l2_norm_output_shape": l2_norm_output_shape,
        "stack_fn": stack_fn,
        "expand_dims_fn": expand_dims_fn,
        "cast_fn": cast_fn,
        "weighted_sum_fn": weighted_sum_fn,
        "compute_color_stability_map": compute_color_stability_map,
        "compute_fft_magnitude_map": compute_fft_magnitude_map,
    }


def save_model(model: tf.keras.Model, path: str) -> None:
    """Save the full Keras model (architecture + weights) to *path*."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    model.save(path)
    logger.info("Model saved → %s", path)


def download_model_weights(url: str, destination_path: str) -> str:
    """Download model weights file from a URL to destination_path."""
    import urllib.request
    os.makedirs(os.path.dirname(destination_path) or ".", exist_ok=True)
    logger.info("Downloading model weights from %s to %s...", url, destination_path)
    urllib.request.urlretrieve(url, destination_path)
    logger.info("Download complete: %s (size: %.2f MB)", destination_path, os.path.getsize(destination_path) / (1024 * 1024))
    return destination_path


def load_model(path: str) -> tf.keras.Model:
    """Load a previously saved FDCS-Net V4 Keras model from *path*."""
    logger.info("Loading model from %s", path)
    custom_objs = get_custom_objects()
    try:
        return tf.keras.models.load_model(path, custom_objects=custom_objs, safe_mode=False)
    except TypeError:
        return tf.keras.models.load_model(path, custom_objects=custom_objs)


def resolve_and_load_model(model_path: str | None = None, auto_download: bool = True) -> tf.keras.Model:
    """
    Resolve model path and load FDCS-Net V4.
    If model_path is None, checks:
      1. Default path 'models/fdcsnet_v4_final.keras'
      2. Environment variable 'FDCSNET_MODEL_URL' for automatic download
    """
    from pathlib import Path
    root_dir = Path(__file__).resolve().parent.parent

    target_path = None
    if model_path:
        target_path = Path(model_path)
        if not target_path.is_absolute():
            target_path = root_dir / target_path
    else:
        target_path = root_dir / "models" / "fdcsnet_v4_final.keras"

    if not target_path.exists():
        env_url = os.environ.get("FDCSNET_MODEL_URL", "").strip()
        if env_url and auto_download:
            try:
                download_model_weights(env_url, str(target_path))
            except Exception as e:
                raise RuntimeError(
                    f"Failed to download model weights from {env_url}: {str(e)}"
                ) from e
        else:
            # Check if any .keras file exists in models/
            models_dir = root_dir / "models"
            keras_files = list(models_dir.glob("*.keras")) if models_dir.exists() else []
            if keras_files:
                target_path = keras_files[0]
                logger.info("Using alternative model file found in models/: %s", target_path)
            else:
                raise FileNotFoundError(
                    f"Model weights not found at '{target_path}'.\n"
                    f"Please place 'fdcsnet_v4_final.keras' into the 'models/' directory, "
                    f"or set the 'FDCSNET_MODEL_URL' environment variable to download it."
                )

    return load_model(str(target_path))


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def get_predictions(model: tf.keras.Model, dataset: tf.data.Dataset) -> tuple:
    """
    Run inference on *dataset* and return (predictions, true_labels).

    Works for both single-output and multi-output models (main_output only).
    """
    all_preds, all_labels = [], []
    for images, labels in dataset:
        outputs = model(images, training=False)
        preds = outputs[0] if isinstance(outputs, (list, tuple)) else outputs
        all_preds.extend(preds.numpy().flatten())
        all_labels.extend(labels.numpy().flatten())
    return np.array(all_preds), np.array(all_labels)


def find_optimal_threshold(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Find the classification threshold that maximises (TPR - FPR) on the ROC
    curve (Youden's J statistic).
    """
    from sklearn.metrics import roc_curve

    fpr, tpr, thresholds = roc_curve(y_true, y_pred)
    optimal_idx = np.argmax(tpr - fpr)
    return float(thresholds[optimal_idx])


def evaluate_model(
    model: tf.keras.Model,
    dataset: tf.data.Dataset,
    threshold: float | None = None,
    val_dataset: tf.data.Dataset | None = None,
) -> dict:
    """
    Evaluate *model* on *dataset* (intended to be the TEST set) and return a
    metrics dictionary.

    Threshold selection methodology
    --------------------------------
    The classification threshold must be chosen using the VALIDATION set,
    never the test set, to avoid leaking test information into the decision
    boundary. Pass one of:
      - `threshold`     : a fixed value (e.g. CLASSIFICATION_THRESHOLD from
                           configs/config.py) to skip selection entirely, or
      - `val_dataset`    : a validation ``tf.data.Dataset`` — the optimal
                           threshold is computed on it via Youden's J
                           statistic and then applied to `dataset` (test).
    If neither is given, this raises an error rather than silently picking
    the threshold on the test set itself.

    Returns
    -------
    dict with keys: accuracy, precision, recall, f1, auc_roc, threshold,
                    confusion_matrix, classification_report
    """
    from sklearn.metrics import (
        classification_report,
        confusion_matrix,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    if threshold is None:
        if val_dataset is None:
            raise ValueError(
                "evaluate_model: no threshold provided and no val_dataset "
                "given. Pass a fixed `threshold` (e.g. CLASSIFICATION_"
                "THRESHOLD from configs/config.py) or a `val_dataset` to "
                "select the threshold on — never select it on the test set."
            )
        y_val_pred, y_val_true = get_predictions(model, val_dataset)
        threshold = find_optimal_threshold(y_val_true, y_val_pred)
        logger.info("Optimal threshold selected on validation set: %.4f", threshold)

    y_pred, y_true = get_predictions(model, dataset)

    y_pred_bin = (y_pred >= threshold).astype(int)
    auc = roc_auc_score(y_true, y_pred)

    results = {
        "threshold": threshold,
        "accuracy": float(np.mean(y_pred_bin == y_true.astype(int))),
        "precision": precision_score(y_true.astype(int), y_pred_bin, zero_division=0),
        "recall": recall_score(y_true.astype(int), y_pred_bin, zero_division=0),
        "f1": f1_score(y_true.astype(int), y_pred_bin, zero_division=0),
        "auc_roc": auc,
        "confusion_matrix": confusion_matrix(y_true.astype(int), y_pred_bin),
        "classification_report": classification_report(
            y_true.astype(int), y_pred_bin, target_names=["Real", "AI"]
        ),
    }
    return results


def analyse_attention_weights(
    model: tf.keras.Model, dataset: tf.data.Dataset
) -> dict:
    """
    Extract and summarise the attention weights from the fusion layer.

    Returns
    -------
    dict with per-branch mean and std of attention weights.
    """
    attn_model = tf.keras.Model(
        inputs=model.input,
        outputs=model.get_layer("attention_weights").output,
    )
    all_attn = []
    for images, _ in dataset:
        aw = attn_model(images, training=False).numpy()
        all_attn.extend(aw)
    all_attn = np.array(all_attn)

    return {
        "spatial": {"mean": float(all_attn[:, 0].mean()), "std": float(all_attn[:, 0].std())},
        "color":   {"mean": float(all_attn[:, 1].mean()), "std": float(all_attn[:, 1].std())},
        "freq":    {"mean": float(all_attn[:, 2].mean()), "std": float(all_attn[:, 2].std())},
    }


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def plot_training_history(histories: list, save_path: str | None = None) -> None:
    """
    Plot combined loss and accuracy curves across multiple training histories.

    Parameters
    ----------
    histories : list of keras.callbacks.History
        One per training stage.
    save_path : str | None
        If provided, the figure is saved to this path.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    all_loss, all_val_loss = [], []
    all_acc, all_val_acc = [], []
    stage_boundaries = []

    for hist in histories:
        all_loss.extend(hist.history["loss"])
        all_val_loss.extend(hist.history["val_loss"])
        stage_boundaries.append(len(all_loss))

        acc_keys = [k for k in hist.history if "accuracy" in k and "val" not in k]
        val_acc_keys = [k for k in hist.history if "accuracy" in k and "val" in k]
        if acc_keys:
            all_acc.extend(hist.history[acc_keys[0]])
        if val_acc_keys:
            all_val_acc.extend(hist.history[val_acc_keys[0]])

    epochs = range(1, len(all_loss) + 1)

    axes[0].plot(epochs, all_loss, "b-", label="Train Loss")
    axes[0].plot(epochs, all_val_loss, "r-", label="Val Loss")
    axes[0].set_title("Loss Across All Stages")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    if all_acc:
        axes[1].plot(range(1, len(all_acc) + 1), all_acc, "b-", label="Train Acc")
    if all_val_acc:
        axes[1].plot(range(1, len(all_val_acc) + 1), all_val_acc, "r-", label="Val Acc")
    axes[1].set_title("Accuracy Across All Stages")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()

    for ax in axes:
        for boundary in stage_boundaries[:-1]:
            ax.axvline(x=boundary, color="gray", linestyle="--", alpha=0.7)

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        plt.savefig(save_path, dpi=150)
        logger.info("Training curves saved → %s", save_path)
    plt.show()


def extract_intermediate_maps(image_input) -> dict:
    """
    Extract intermediate FFT magnitude map and Color stability map for visual explainability.
    
    Returns
    -------
    dict with 'fft_magnitude' and 'color_stability' as (128, 128, 3) float32 numpy arrays in [0, 1].
    """
    from data_preprocessing import preprocess_image
    from model import compute_color_stability_map, compute_fft_magnitude_map

    img_tensor = preprocess_image(image_input)
    color_map = compute_color_stability_map(img_tensor)
    fft_map = compute_fft_magnitude_map(img_tensor)

    return {
        "color_stability": np.clip(np.squeeze(color_map.numpy()), 0.0, 1.0),
        "fft_magnitude": np.clip(np.squeeze(fft_map.numpy()), 0.0, 1.0),
    }


def extract_attention_weights(model: tf.keras.Model, img_tensor: tf.Tensor) -> dict | None:
    """Extract softmax attention weights from the fusion layer for a single image tensor."""
    try:
        if "attention_weights" in [layer.name for layer in model.layers]:
            attn_layer = model.get_layer("attention_weights")
            attn_model = tf.keras.Model(inputs=model.input, outputs=attn_layer.output)
            raw_weights = attn_model(img_tensor, training=False).numpy().flatten()
            return {
                "spatial": float(raw_weights[0]),
                "color": float(raw_weights[1]),
                "frequency": float(raw_weights[2]),
            }
    except Exception as e:
        logger.debug("Could not extract attention weights: %s", e)
    return None


def predict_single_image(
    model: tf.keras.Model, image_input, threshold: float = CLASSIFICATION_THRESHOLD
) -> dict:
    """
    Run inference on an image (file path, bytes, BytesIO, or PIL Image).

    Parameters
    ----------
    model       : Loaded FDCS-Net V4 model.
    image_input : File path string, Path, bytes, BytesIO, or PIL Image.
    threshold   : Classification threshold (default: CLASSIFICATION_THRESHOLD from config).

    Returns
    -------
    dict with keys: prediction (float), confidence (float), label (str),
                    threshold (float), attention_weights (dict or None)
    """
    from data_preprocessing import preprocess_image

    img_tensor = preprocess_image(image_input)

    outputs = model(img_tensor, training=False)
    main_out = outputs[0] if isinstance(outputs, (list, tuple)) else outputs
    prob = float(np.squeeze(main_out.numpy()))
    # Clamp probability to [0.0, 1.0] for safety
    prob = max(0.0, min(1.0, prob))

    attn_weights = extract_attention_weights(model, img_tensor)

    return {
        "prediction": prob,
        "confidence": max(prob, 1.0 - prob),
        "label": "AI-Generated" if prob >= threshold else "Real",
        "threshold": float(threshold),
        "attention_weights": attn_weights,
    }