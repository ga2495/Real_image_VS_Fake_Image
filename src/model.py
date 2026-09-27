"""
model.py
========
FDCS-Net V4 architecture definition.

Key architectural components:
  - Spatial Branch  : EfficientNetB0 → GAP → Dense(256) → BN → ReLU
  - Color Branch    : Color Stability Map → CNN → GAP → Dense(256) → BN
  - FFT Branch      : FFT Magnitude Map   → CNN → GAP → Dense(256) → BN
  - Fusion          : L2-normalise all → Attention-weighted sum
  - Auxiliary heads : Separate classifier for each branch (training only)
"""

import logging

import keras
import tensorflow as tf
from tensorflow.keras import layers, Model, regularizers
from tensorflow.keras.applications import EfficientNetB0

from configs.config import (
    IMG_SIZE,
    COLOR_MAP_SIZE,
    FEATURE_DIM,
    WEIGHT_DECAY,
    NOISE_SIGMA,
    QUANT_BITS,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Feature extraction functions (Keras-serialisable)
# ---------------------------------------------------------------------------

@keras.saving.register_keras_serializable()
def compute_fft_magnitude_map(images: tf.Tensor) -> tf.Tensor:
    """
    Compute 2-D FFT magnitude spectrum as a spatial feature map.

    The result is log-scaled, normalised, resized to *COLOR_MAP_SIZE*, and
    replicated across 3 channels so it can be processed by a standard CNN.
    """
    original_dtype = images.dtype
    images_f32 = tf.cast(images, tf.float32)

    color_weights = tf.constant([0.2989, 0.5870, 0.1140], dtype=tf.float32)
    gray = tf.reduce_sum(images_f32 * color_weights, axis=-1)

    fft = tf.signal.fft2d(tf.cast(gray, tf.complex64))
    fft_shifted = tf.signal.fftshift(fft, axes=[1, 2])
    magnitude = tf.abs(fft_shifted)
    log_mag = tf.math.log(magnitude + 1.0)

    batch_min = tf.reduce_min(log_mag, axis=[1, 2], keepdims=True)
    batch_max = tf.reduce_max(log_mag, axis=[1, 2], keepdims=True)
    log_mag = (log_mag - batch_min) / (batch_max - batch_min + 1e-8)

    log_mag = tf.expand_dims(log_mag, -1)
    log_mag = tf.image.resize(log_mag, [COLOR_MAP_SIZE, COLOR_MAP_SIZE])
    log_mag = tf.repeat(log_mag, 3, axis=-1)
    return tf.cast(log_mag, original_dtype)


@keras.saving.register_keras_serializable()
def compute_color_stability_map(images: tf.Tensor) -> tf.Tensor:
    """
    Compute a perturbation-based colour-difference map.

    Applies Gaussian noise + quantisation + local averaging to the input,
    then measures the per-pixel absolute difference from the original.
    AI-generated images typically show distinctive patterns in this map.
    """
    original_dtype = images.dtype
    images_f32 = tf.cast(images, tf.float32)

    noise = tf.random.normal(
        tf.shape(images_f32), mean=0.0, stddev=NOISE_SIGMA, dtype=tf.float32
    )
    noisy = tf.clip_by_value(images_f32 + noise, 0.0, 1.0)

    num_levels = tf.constant(2.0 ** QUANT_BITS, dtype=tf.float32)
    quantized = tf.round(noisy * (num_levels - 1.0)) / (num_levels - 1.0)

    avg_kernel = tf.ones([3, 3, 3, 1], dtype=tf.float32) / 9.0
    restored = tf.nn.depthwise_conv2d(quantized, avg_kernel, [1, 1, 1, 1], "SAME")

    diff = tf.abs(images_f32 - restored)
    batch_min = tf.reduce_min(diff, axis=[1, 2, 3], keepdims=True)
    batch_max = tf.reduce_max(diff, axis=[1, 2, 3], keepdims=True)
    diff = (diff - batch_min) / (batch_max - batch_min + 1e-8)

    result = tf.image.resize(diff, [COLOR_MAP_SIZE, COLOR_MAP_SIZE])
    return tf.cast(result, original_dtype)


# ---------------------------------------------------------------------------
# Lambda helpers (must be named for Keras serialisation)
# ---------------------------------------------------------------------------

@keras.saving.register_keras_serializable()
def preprocess_fn(x):
    """EfficientNet preprocessing (scales [0,1] input to [-1, 255] range)."""
    return tf.keras.applications.efficientnet.preprocess_input(x * 255.0)


@keras.saving.register_keras_serializable()
def l2_norm_fn(x):
    """L2-normalise along the feature dimension."""
    return tf.cast(
        tf.math.l2_normalize(tf.cast(x, tf.float32), axis=-1), x.dtype
    )


@keras.saving.register_keras_serializable()
def l2_norm_output_shape(input_shape):
    """Static output-shape function for l2_norm_fn (shape-preserving)."""
    return input_shape


@keras.saving.register_keras_serializable()
def stack_fn(x):
    """Stack a list of tensors along axis=1."""
    return tf.stack(x, axis=1)


@keras.saving.register_keras_serializable()
def expand_dims_fn(x):
    """Expand dims along last axis."""
    return tf.expand_dims(x, -1)


@keras.saving.register_keras_serializable()
def cast_fn(x):
    """Cast tensor to float32."""
    return tf.cast(x, tf.float32)


@keras.saving.register_keras_serializable()
def weighted_sum_fn(inputs):
    """Compute attention-weighted sum: sum(features * weights, axis=1)."""
    return tf.reduce_sum(inputs[0] * inputs[1], axis=1)


# ---------------------------------------------------------------------------
# Branch builders
# ---------------------------------------------------------------------------

def build_cnn_branch(input_shape: tuple, name_prefix: str) -> Model:
    """
    Build a lightweight CNN branch for processing spatial feature maps.

    Architecture:
        Conv2D(32) → ReLU → MaxPool →
        Conv2D(64) → ReLU → MaxPool →
        GlobalAveragePooling → Dense(FEATURE_DIM) → BatchNorm
    """
    inp = layers.Input(shape=input_shape, name=f"{name_prefix}_input")
    x = layers.Conv2D(
        32, (3, 3), padding="same",
        kernel_regularizer=regularizers.l2(WEIGHT_DECAY),
        name=f"{name_prefix}_conv1",
    )(inp)
    x = layers.ReLU(name=f"{name_prefix}_relu1")(x)
    x = layers.MaxPooling2D((2, 2), name=f"{name_prefix}_pool1")(x)

    x = layers.Conv2D(
        64, (3, 3), padding="same",
        kernel_regularizer=regularizers.l2(WEIGHT_DECAY),
        name=f"{name_prefix}_conv2",
    )(x)
    x = layers.ReLU(name=f"{name_prefix}_relu2")(x)
    x = layers.MaxPooling2D((2, 2), name=f"{name_prefix}_pool2")(x)

    x = layers.GlobalAveragePooling2D(name=f"{name_prefix}_gap")(x)
    x = layers.Dense(
        FEATURE_DIM,
        kernel_regularizer=regularizers.l2(WEIGHT_DECAY),
        name=f"{name_prefix}_dense",
    )(x)
    x = layers.BatchNormalization(name=f"{name_prefix}_bn")(x)
    return Model(inputs=inp, outputs=x, name=f"{name_prefix}_branch")


# ---------------------------------------------------------------------------
# Full FDCS-Net V4
# ---------------------------------------------------------------------------

def build_fdcsnet_v4(
    freeze_ratio: float = 0.90,
    enable_aux: bool = True,
    freeze_backbone_completely: bool = False,
) -> tuple:
    """
    Build the FDCS-Net V4 hybrid model with attention-based fusion.

    Parameters
    ----------
    freeze_ratio : float
        Fraction of EfficientNet layers to freeze (0.0 = all trainable).
    enable_aux : bool
        If True, attach auxiliary classifier heads to color and FFT branches.
    freeze_backbone_completely : bool
        If True, freeze the entire EfficientNet backbone regardless of
        *freeze_ratio*.

    Returns
    -------
    model          : tf.keras.Model  – Full FDCS-Net V4
    backbone       : tf.keras.Model  – EfficientNetB0 sub-model
    image_branch   : tf.keras.Model  – Spatial branch sub-model
    color_branch   : tf.keras.Model  – Color stability branch
    freq_branch    : tf.keras.Model  – FFT magnitude branch
    """
    raw_input = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3), name="raw_input")

    # ── Branch 1 : Spatial (EfficientNetB0) ──────────────────────────────
    img_in = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3), name="img_branch_input")
    x_img = layers.Lambda(
        preprocess_fn,
        output_shape=(IMG_SIZE, IMG_SIZE, 3),
        name="efficientnet_preprocess",
    )(img_in)

    backbone = EfficientNetB0(
        include_top=False, weights="imagenet",
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
    )
    if freeze_backbone_completely:
        for layer in backbone.layers:
            layer.trainable = False
    else:
        freeze_until = int(len(backbone.layers) * freeze_ratio)
        for i, layer in enumerate(backbone.layers):
            layer.trainable = i >= freeze_until

    x_img = backbone(x_img)
    x_img = layers.GlobalAveragePooling2D(name="img_gap")(x_img)
    x_img = layers.Dense(
        FEATURE_DIM,
        kernel_regularizer=regularizers.l2(WEIGHT_DECAY),
        name="img_dense",
    )(x_img)
    x_img = layers.BatchNormalization(name="img_bn")(x_img)
    x_img = layers.ReLU(name="img_relu")(x_img)
    image_branch = Model(inputs=img_in, outputs=x_img, name="SpatialBranch")
    F_spatial = image_branch(raw_input)

    # ── Branch 2 : Color Stability ────────────────────────────────────────
    color_map = layers.Lambda(
        compute_color_stability_map,
        output_shape=(COLOR_MAP_SIZE, COLOR_MAP_SIZE, 3),
        name="lambda_color_stability",
    )(raw_input)
    color_branch = build_cnn_branch((COLOR_MAP_SIZE, COLOR_MAP_SIZE, 3), "color")
    F_color = color_branch(color_map)

    # ── Branch 3 : FFT Magnitude ──────────────────────────────────────────
    fft_map = layers.Lambda(
        compute_fft_magnitude_map,
        output_shape=(COLOR_MAP_SIZE, COLOR_MAP_SIZE, 3),
        name="lambda_fft_magnitude",
    )(raw_input)
    freq_branch = build_cnn_branch((COLOR_MAP_SIZE, COLOR_MAP_SIZE, 3), "freq")
    F_freq = freq_branch(fft_map)

    # ── L2 Normalisation ──────────────────────────────────────────────────
    F_spatial_norm = layers.Lambda(
        l2_norm_fn, output_shape=l2_norm_output_shape, name="norm_spatial"
    )(F_spatial)
    F_color_norm = layers.Lambda(
        l2_norm_fn, output_shape=l2_norm_output_shape, name="norm_color"
    )(F_color)
    F_freq_norm = layers.Lambda(
        l2_norm_fn, output_shape=l2_norm_output_shape, name="norm_freq"
    )(F_freq)

    # ── Attention Fusion ──────────────────────────────────────────────────
    stacked = layers.Lambda(stack_fn, name="stack_features")(
        [F_spatial_norm, F_color_norm, F_freq_norm]
    )
    concat_for_attn = layers.Concatenate(name="concat_for_attn")(
        [F_spatial_norm, F_color_norm, F_freq_norm]
    )
    attn_weights = layers.Dense(
        3,
        activation="softmax",
        dtype="float32",
        kernel_regularizer=regularizers.l2(WEIGHT_DECAY),
        name="attention_weights",
    )(concat_for_attn)
    attn_weights_expanded = layers.Lambda(
        expand_dims_fn, name="expand_attn_weights"
    )(attn_weights)
    stacked_f32 = layers.Lambda(cast_fn, name="cast_stacked")(stacked)
    F_final = layers.Lambda(weighted_sum_fn, name="weighted_sum")(
        [stacked_f32, attn_weights_expanded]
    )

    # ── Main Classifier Head ──────────────────────────────────────────────
    x = layers.Dense(
        128,
        kernel_regularizer=regularizers.l2(WEIGHT_DECAY),
        name="head_dense1",
    )(F_final)
    x = layers.ReLU(name="head_relu1")(x)
    x = layers.Dropout(0.3, name="head_drop1")(x)
    main_output = layers.Dense(
        1, activation="sigmoid", dtype="float32", name="main_output"
    )(x)

    outputs = [main_output]

    # ── Auxiliary Heads (training only) ───────────────────────────────────
    if enable_aux:
        ca = layers.Dense(
            128, kernel_regularizer=regularizers.l2(WEIGHT_DECAY),
            name="color_aux_dense",
        )(F_color_norm)
        ca = layers.ReLU(name="color_aux_relu")(ca)
        ca = layers.Dropout(0.3, name="color_aux_drop")(ca)
        color_output = layers.Dense(
            1, activation="sigmoid", dtype="float32", name="color_output"
        )(ca)

        fa = layers.Dense(
            128, kernel_regularizer=regularizers.l2(WEIGHT_DECAY),
            name="freq_aux_dense",
        )(F_freq_norm)
        fa = layers.ReLU(name="freq_aux_relu")(fa)
        fa = layers.Dropout(0.3, name="freq_aux_drop")(fa)
        freq_output = layers.Dense(
            1, activation="sigmoid", dtype="float32", name="freq_output"
        )(fa)

        outputs = [main_output, color_output, freq_output]

    model = Model(inputs=raw_input, outputs=outputs, name="FDCSNet_v4")
    logger.info(
        "Built FDCSNet_v4 | enable_aux=%s | trainable_params=%d",
        enable_aux,
        model.count_params(),
    )
    return model, backbone, image_branch, color_branch, freq_branch