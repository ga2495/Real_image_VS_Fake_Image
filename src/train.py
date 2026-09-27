"""
train.py
========
End-to-end 3-stage training script for FDCS-Net V4.

Usage
-----
    python src/train.py [--config configs/config.py] [--output models/]

Stages
------
  Stage 1 – Train EfficientNet spatial branch alone (backbone 90% frozen).
  Stage 2 – Freeze spatial branch; train Color + FFT branches with auxiliary
             supervision.
  Stage 3 – Unfreeze all; end-to-end fine-tuning with attention fusion and
             auxiliary losses.
"""

import argparse
import gc
import logging
import os
import time

import tensorflow as tf
import keras

from data_preprocessing import (
    build_dataset,
    adapt_dataset_for_multi_output,
)
from model import build_fdcsnet_v4
from utils import (
    get_callbacks,
    compile_model_single,
    compile_model_multi,
    save_model,
    plot_training_history,
    set_mixed_precision,
)
from configs.config import (
    TRAIN_DIR,
    TEST_DIR,
    MODEL_SAVE_DIR,
    VALIDATION_SPLIT,
    BATCH_SIZE,
    LEARNING_RATE,
    STAGE1_EPOCHS,
    STAGE1_FREEZE_RATIO,
    STAGE2_EPOCHS,
    STAGE3_EPOCHS,
    STAGE3_FREEZE_RATIO,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s – %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Training stages
# ---------------------------------------------------------------------------

def stage1(train_dataset, val_dataset, steps_per_epoch, validation_steps):
    """Train EfficientNet spatial branch alone."""
    logger.info("=" * 60)
    logger.info("STAGE 1 – Train EfficientNet spatial branch alone")
    logger.info("=" * 60)

    model, backbone, image_branch, color_branch, freq_branch = build_fdcsnet_v4(
        freeze_ratio=STAGE1_FREEZE_RATIO,
        enable_aux=False,
        freeze_backbone_completely=False,
    )
    color_branch.trainable = False
    freq_branch.trainable = False
    compile_model_single(model, lr=LEARNING_RATE)

    t0 = time.time()
    history = model.fit(
        train_dataset,
        steps_per_epoch=steps_per_epoch,
        validation_data=val_dataset,
        validation_steps=validation_steps,
        epochs=STAGE1_EPOCHS,
        callbacks=get_callbacks(),
        verbose=1,
    )
    elapsed = time.time() - t0
    logger.info("Stage 1 done: %d epochs in %.1f min", len(history.history["loss"]), elapsed / 60)

    weights_path = os.path.join(MODEL_SAVE_DIR, "stage1_weights.weights.h5")
    model.save_weights(weights_path)
    logger.info("Stage 1 weights saved → %s", weights_path)
    return history, weights_path


def stage2(
    train_multi, val_multi, steps_per_epoch, validation_steps, stage1_weights_path
):
    """Train Color + FFT branches (EfficientNet frozen)."""
    logger.info("=" * 60)
    logger.info("STAGE 2 – Train Color + FFT branches (EfficientNet frozen)")
    logger.info("=" * 60)

    tf.keras.backend.clear_session()
    gc.collect()

    model, backbone, image_branch, color_branch, freq_branch = build_fdcsnet_v4(
        freeze_ratio=1.0,
        enable_aux=True,
        freeze_backbone_completely=True,
    )
    model.load_weights(stage1_weights_path, skip_mismatch=True)
    image_branch.trainable = False
    color_branch.trainable = True
    freq_branch.trainable = True
    compile_model_multi(model, lr=LEARNING_RATE)

    t0 = time.time()
    history = model.fit(
        train_multi,
        steps_per_epoch=steps_per_epoch,
        validation_data=val_multi,
        validation_steps=validation_steps,
        epochs=STAGE2_EPOCHS,
        callbacks=get_callbacks(),
        verbose=1,
    )
    elapsed = time.time() - t0
    logger.info("Stage 2 done: %d epochs in %.1f min", len(history.history["loss"]), elapsed / 60)

    weights_path = os.path.join(MODEL_SAVE_DIR, "stage2_weights.weights.h5")
    model.save_weights(weights_path)
    logger.info("Stage 2 weights saved → %s", weights_path)
    return history, weights_path


def stage3(
    train_multi, val_multi, steps_per_epoch, validation_steps, stage2_weights_path
):
    """Full end-to-end fine-tuning with attention fusion."""
    logger.info("=" * 60)
    logger.info("STAGE 3 – Full end-to-end training with attention fusion")
    logger.info("=" * 60)

    tf.keras.backend.clear_session()
    gc.collect()

    model, backbone, image_branch, color_branch, freq_branch = build_fdcsnet_v4(
        freeze_ratio=STAGE3_FREEZE_RATIO,
        enable_aux=True,
        freeze_backbone_completely=False,
    )
    model.load_weights(stage2_weights_path, skip_mismatch=True)

    image_branch.trainable = True
    color_branch.trainable = True
    freq_branch.trainable = True
    freeze_until = int(len(backbone.layers) * STAGE3_FREEZE_RATIO)
    for i, layer in enumerate(backbone.layers):
        layer.trainable = i >= freeze_until

    compile_model_multi(model, lr=LEARNING_RATE * 0.3)

    t0 = time.time()
    history = model.fit(
        train_multi,
        steps_per_epoch=steps_per_epoch,
        validation_data=val_multi,
        validation_steps=validation_steps,
        epochs=STAGE3_EPOCHS,
        callbacks=get_callbacks(),
        verbose=1,
    )
    elapsed = time.time() - t0
    logger.info("Stage 3 done: %d epochs in %.1f min", len(history.history["loss"]), elapsed / 60)

    final_weights_path = os.path.join(MODEL_SAVE_DIR, "fdcsnet_v4_final.weights.h5")
    model_save_path = os.path.join(MODEL_SAVE_DIR, "fdcsnet_v4_final.keras")
    model.save_weights(final_weights_path)
    save_model(model, model_save_path)
    logger.info("Final model saved → %s", model_save_path)
    return history, model, model_save_path


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Train FDCS-Net V4")
    parser.add_argument(
        "--train-dir", default=TRAIN_DIR,
        help="Path to training dataset root (default: from config)"
    )
    parser.add_argument(
        "--output-dir", default=MODEL_SAVE_DIR,
        help="Directory to save model checkpoints (default: from config)"
    )
    parser.add_argument(
        "--no-mixed-precision", action="store_true",
        help="Disable mixed precision training"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    # Override the config default so all stages save into the requested dir.
    global MODEL_SAVE_DIR
    MODEL_SAVE_DIR = args.output_dir

    if not args.no_mixed_precision:
        set_mixed_precision()

    # ── Build datasets ────────────────────────────────────────────────────
    logger.info("Loading datasets from: %s", args.train_dir)
    train_dataset, n_train = build_dataset(
        args.train_dir, is_training=True, subset="training"
    )
    val_dataset, n_val = build_dataset(
        args.train_dir, is_training=False, subset="validation"
    )
    steps_per_epoch = n_train // BATCH_SIZE
    validation_steps = n_val // BATCH_SIZE
    logger.info("Train: %d samples (%d steps/epoch)", n_train, steps_per_epoch)
    logger.info("Val:   %d samples (%d val steps)", n_val, validation_steps)

    train_multi = adapt_dataset_for_multi_output(train_dataset)
    val_multi = adapt_dataset_for_multi_output(val_dataset)

    # ── 3-Stage Training ──────────────────────────────────────────────────
    total_start = time.time()

    h1, s1_weights = stage1(train_dataset, val_dataset, steps_per_epoch, validation_steps)
    h2, s2_weights = stage2(train_multi, val_multi, steps_per_epoch, validation_steps, s1_weights)
    h3, final_model, model_path = stage3(train_multi, val_multi, steps_per_epoch, validation_steps, s2_weights)

    total_time = (time.time() - total_start) / 60
    logger.info("Total training time: %.1f min", total_time)

    # ── Plot training curves ──────────────────────────────────────────────
    plot_path = os.path.join(args.output_dir, "training_curves.png")
    plot_training_history([h1, h2, h3], save_path=plot_path)
    logger.info("Training curves saved → %s", plot_path)

    logger.info("Training complete. Model saved at: %s", model_path)


if __name__ == "__main__":
    main()