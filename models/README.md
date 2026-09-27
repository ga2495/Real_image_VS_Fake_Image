# SynthLens — FDCS-Net V4 Model Artifacts

This directory holds the trained weights and serialization artifacts for **FDCS-Net V4** (Hybrid Spatial, Frequency, and Color Stability AI Image Detector).

---

## Expected Artifact

- **Filename:** `fdcsnet_v4_final.keras`
- **Format:** Keras 3 SavedModel (`.keras` format)
- **Input shape:** `(None, 256, 256, 3)`
- **Output:** Sigmoid scalar in `[0.0, 1.0]` representing `P(AI-Generated)`
- **Model Architecture:** FDCS-Net V4
  - **Spatial Branch:** EfficientNet-B0 (ImageNet pretrained backbone) → Dense(256) → BatchNorm → L2 Normalization
  - **Color Stability Branch:** Perturbation Map (`compute_color_stability_map`) → CNN(32→64) → Dense(256) → BatchNorm → L2 Normalization
  - **Frequency Branch:** 2-D FFT Log-Magnitude (`compute_fft_magnitude_map`) → CNN(32→64) → Dense(256) → BatchNorm → L2 Normalization
  - **Attention Fusion:** Softmax attention weights across all 3 normalized branches → Weighted Sum
  - **Classifier Head:** Dense(128) → ReLU → Dropout(0.3) → Dense(1, Sigmoid)

---

## Model Availability

The model artifact `fdcsnet_v4_final.keras` (~18.9 MB) is bundled directly within this repository. No external downloading or third-party hosting credentials are required for local execution or Streamlit Community Cloud deployment.

### Fallback Options
1. **Retraining:** If training from scratch with `src/train.py`, output artifacts will save to `models/fdcsnet_v4_final.keras`.
2. **Optional Remote Override:** If a different model URL is specified via the `FDCSNET_MODEL_URL` environment variable, the application can optionally download new weights at runtime.

### 3. Generate from Training Pipeline
If you have the training dataset prepared under `data/train/` (with `real/` and `ai/` subfolders), run the full 3-stage training pipeline:
```bash
python src/train.py --output-dir models/
```

---

## Classification Threshold

- **Validation-selected threshold:** `0.80` (configured in `configs/config.py`)
- **Positive class (1.0):** AI-Generated
- **Negative class (0.0):** Real photograph
