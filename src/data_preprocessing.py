"""
data_preprocessing.py
=====================
Data loading, augmentation, and pipeline construction for FDCS-Net V4.

Dataset directory structure expected:
    data/
    ├── train/
    │   ├── real/
    │   └── ai/
    └── test/
        ├── real/
        └── ai/
"""

import os
import io
import logging
from pathlib import Path
from PIL import Image

import numpy as np
import tensorflow as tf

from configs.config import (
    IMG_SIZE,
    BATCH_SIZE,
    VALIDATION_SPLIT,
    NOISE_SIGMA,
    QUANT_BITS,
    COLOR_MAP_SIZE,
    SUPPORTED_EXTENSIONS,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Image I/O, validation & unified preprocessing
# ---------------------------------------------------------------------------

def validate_image(image_input, max_size_mb: float = 15.0) -> tuple[bool, str]:
    """
    Validate uploaded image input.
    Returns (is_valid, error_message).
    """
    try:
        if isinstance(image_input, (str, Path)):
            path = Path(image_input)
            if not path.exists():
                return False, f"File does not exist: {path}"
            size_mb = path.stat().st_size / (1024 * 1024)
            if size_mb > max_size_mb:
                return False, f"Image size ({size_mb:.1f} MB) exceeds limit ({max_size_mb} MB)."
            with Image.open(path) as img:
                img.verify()
            with Image.open(path) as img:
                fmt = (img.format or "").upper()
                if fmt not in ["JPEG", "JPG", "PNG", "WEBP", "BMP"]:
                    return False, f"Unsupported format '{fmt}'. Allowed: JPG, JPEG, PNG, WEBP, BMP."
            return True, ""
        elif isinstance(image_input, (bytes, bytearray, io.BytesIO)):
            raw_bytes = image_input.getvalue() if isinstance(image_input, io.BytesIO) else image_input
            size_mb = len(raw_bytes) / (1024 * 1024)
            if size_mb > max_size_mb:
                return False, f"Image size ({size_mb:.1f} MB) exceeds limit ({max_size_mb} MB)."
            with Image.open(io.BytesIO(raw_bytes)) as img:
                img.verify()
            with Image.open(io.BytesIO(raw_bytes)) as img:
                fmt = (img.format or "").upper()
                if fmt not in ["JPEG", "JPG", "PNG", "WEBP", "BMP"]:
                    return False, f"Unsupported format '{fmt}'. Allowed: JPG, JPEG, PNG, WEBP, BMP."
            return True, ""
        elif isinstance(image_input, Image.Image):
            return True, ""
        elif isinstance(image_input, np.ndarray):
            return True, ""
        elif hasattr(image_input, "read"):
            pos = image_input.tell() if hasattr(image_input, "tell") else 0
            content = image_input.read()
            if hasattr(image_input, "seek"):
                image_input.seek(pos)
            return validate_image(content, max_size_mb=max_size_mb)
        else:
            return False, f"Unrecognized image input type: {type(image_input)}"
    except Exception as e:
        return False, f"Corrupted or unreadable image file: {str(e)}"


def preprocess_image(image_input) -> tf.Tensor:
    """
    Unified inference preprocessing function for FDCS-Net V4.
    Matches parse_image: Decodes to RGB, resizes to (IMG_SIZE, IMG_SIZE),
    normalizes to [0.0, 1.0], and adds a batch dimension (1, 256, 256, 3).
    """
    if isinstance(image_input, (str, Path)):
        img_bytes = tf.io.read_file(str(image_input))
        img = tf.image.decode_image(img_bytes, channels=3, expand_animations=False)
    elif isinstance(image_input, (bytes, bytearray)):
        img = tf.image.decode_image(image_input, channels=3, expand_animations=False)
    elif isinstance(image_input, io.BytesIO):
        img = tf.image.decode_image(image_input.getvalue(), channels=3, expand_animations=False)
    elif isinstance(image_input, Image.Image):
        img_rgb = image_input.convert("RGB")
        img_np = np.array(img_rgb, dtype=np.float32)
        img = tf.convert_to_tensor(img_np, dtype=tf.float32)
    elif isinstance(image_input, np.ndarray):
        img = tf.convert_to_tensor(image_input, dtype=tf.float32)
        if len(img.shape) == 2:
            img = tf.repeat(tf.expand_dims(img, -1), 3, axis=-1)
    elif hasattr(image_input, "read"):
        pos = image_input.tell() if hasattr(image_input, "tell") else 0
        content = image_input.read()
        if hasattr(image_input, "seek"):
            image_input.seek(pos)
        img = tf.image.decode_image(content, channels=3, expand_animations=False)
    else:
        raise ValueError(f"Unsupported image input type: {type(image_input)}")

    # Ensure 3 channels
    if img.shape[-1] == 1:
        img = tf.repeat(img, 3, axis=-1)
    elif img.shape[-1] == 4:
        img = img[:, :, :3]

    img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE])
    img = tf.cast(img, tf.float32)
    if tf.reduce_max(img) > 1.0:
        img = img / 255.0
    if len(img.shape) == 3:
        img = tf.expand_dims(img, 0)
    return img


def parse_image(file_path: tf.Tensor, label: tf.Tensor):
    """Read a JPEG/PNG from disk, resize, and normalise to [0, 1]."""
    img = tf.io.read_file(file_path)
    img = tf.image.decode_image(img, channels=3, expand_animations=False)
    img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE])
    img = tf.cast(img, tf.float32) / 255.0
    return img, label


def augment_image(image: tf.Tensor, label: tf.Tensor):
    """Apply random photometric augmentation (training only)."""
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_brightness(image, max_delta=0.2)
    image = tf.image.random_contrast(image, lower=0.8, upper=1.2)
    image = tf.image.random_saturation(image, lower=0.8, upper=1.2)
    noise = tf.random.normal(tf.shape(image), mean=0.0, stddev=0.02)
    image = tf.clip_by_value(image + noise, 0.0, 1.0)
    return image, label


# ---------------------------------------------------------------------------
# Dataset builder
# ---------------------------------------------------------------------------

def _collect_file_paths(directory: str):
    """Scan *directory* for images under sub-folders 'real' and 'ai'."""
    file_paths, labels = [], []
    for class_idx, class_name in enumerate(["real", "ai"]):
        class_path = os.path.join(directory, class_name)
        if not os.path.exists(class_path):
            logger.warning("Class directory not found: %s", class_path)
            continue
        for fname in sorted(os.listdir(class_path)):
            if fname.lower().endswith(SUPPORTED_EXTENSIONS):
                file_paths.append(os.path.join(class_path, fname))
                labels.append(float(class_idx))
    return np.array(file_paths), np.array(labels, dtype=np.float32)


def build_dataset(
    directory: str,
    is_training: bool = True,
    validation_split: float = VALIDATION_SPLIT,
    subset: str = "training",
    cache_path: str = "",
) -> tuple:
    """
    Build a ``tf.data.Dataset`` from *directory*.

    Parameters
    ----------
    directory : str
        Root directory containing 'real/' and 'ai/' sub-folders.
    is_training : bool
        If True, apply augmentation and repeat indefinitely.
    validation_split : float
        Fraction of data to reserve for validation.
    subset : str
        ``'training'`` or ``'validation'``.
    cache_path : str
        Path prefix for on-disk caching (empty → in-memory cache).

    Returns
    -------
    dataset : tf.data.Dataset
    n_samples : int
        Number of samples in this split.
    """
    file_paths, labels = _collect_file_paths(directory)
    if len(file_paths) == 0:
        raise FileNotFoundError(f"No images found in {directory}")

    np.random.seed(42)
    indices = np.random.permutation(len(file_paths))
    file_paths, labels = file_paths[indices], labels[indices]

    split_idx = int(len(file_paths) * (1 - validation_split))
    if subset == "training":
        file_paths, labels = file_paths[:split_idx], labels[:split_idx]
    else:
        file_paths, labels = file_paths[split_idx:], labels[split_idx:]

    dataset = tf.data.Dataset.from_tensor_slices((file_paths, labels))
    dataset = dataset.map(parse_image, num_parallel_calls=tf.data.AUTOTUNE)

    if cache_path:
        dataset = dataset.cache(cache_path)
    else:
        dataset = dataset.cache()

    if is_training:
        dataset = dataset.shuffle(buffer_size=min(2000, len(file_paths)))
        dataset = dataset.map(augment_image, num_parallel_calls=tf.data.AUTOTUNE)
        dataset = dataset.repeat()
    else:
        dataset = dataset.repeat()

    dataset = dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    logger.info("Built %s split: %d samples", subset, len(file_paths))
    return dataset, len(file_paths)


def build_test_dataset(directory: str) -> tuple:
    """
    Build a non-repeating test ``tf.data.Dataset``.

    Returns
    -------
    dataset : tf.data.Dataset
    labels : np.ndarray
    """
    file_paths, labels = _collect_file_paths(directory)
    if len(file_paths) == 0:
        raise FileNotFoundError(f"No images found in {directory}")

    dataset = tf.data.Dataset.from_tensor_slices((file_paths, labels))
    dataset = dataset.map(parse_image, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    return dataset, labels


def adapt_dataset_for_multi_output(dataset: tf.data.Dataset) -> tf.data.Dataset:
    """
    Replicate the label for all three model outputs (main, color, freq).
    Required for Stage 2 & Stage 3 multi-output training.
    """
    def _adapt(image, label):
        return image, {
            "main_output": label,
            "color_output": label,
            "freq_output": label,
        }

    return dataset.map(_adapt, num_parallel_calls=tf.data.AUTOTUNE)