# SynthLens: AI Image Detection

A production-ready deep learning framework and interactive Streamlit web application designed to distinguish authentic photographs from AI-generated imagery. SynthLens is powered by **FDCS-Net V4** (Fused Domain Color-Stability Network), a multi-branch architecture that dynamically combines spatial semantics, frequency-domain spectral artifacts, and color-perturbation stability through attention-based feature fusion.

---

## Architecture Overview

```text
                               Input Image (256x256x3)
                                          |
                     +--------------------+--------------------+
                     |                    |                    |
             +---------------+   +-----------------+   +-----------------+
             | Spatial Branch |   |  Color Branch   |   | Frequency Branch|
             | EfficientNet-B0|   | Color-Stability |   |  FFT Magnitude  |
             |  (pretrained)  |   | Map -> CNN      |   |  Map -> CNN     |
             +-------+--------+   +--------+--------+   +--------+--------+
                     |                    |                    |
                     +----------> L2-Normalize each branch <---+
                                          |
                               +----------+----------+
                               |   Attention Fusion  |
                               |  (softmax weights)  |
                               +----------+----------+
                                          |
                               +----------+----------+
                               |   Classifier Head   |
                               | (128 -> 1, sigmoid) |
                               +----------+----------+
                                          |
                                 Real / AI-Generated
```

### Core Architecture Components

- **Spatial Branch:** Leverages an **EfficientNet-B0** convolutional backbone (ImageNet-pretrained) to extract multi-scale semantic representations and spatial texture irregularities.
- **Color Stability Branch:** Constructs a perturbation-difference map using Gaussian noise, color quantization, and local averaging to expose generative color-grid instability, processed via a dedicated lightweight CNN.
- **Frequency Branch:** Computes a log-scaled 2-D Fast Fourier Transform (FFT) magnitude spectrum to capture high-frequency periodic grid artifacts common to GAN and diffusion generative pipelines, processed via a dedicated CNN.
- **Attention Fusion:** Applies L2 normalization across branch embeddings and learns per-image dynamic softmax attention weights rather than naive concatenation.
- **Classifier Head:** Dense(128) with ReLU and Dropout(0.3) followed by a single Sigmoid neuron outputting $P(\text{AI-Generated}) \in [0.0, 1.0]$.

---

## Features

- **Hybrid Multi-Domain Analysis:** Merges spatial, color, and frequency domain cues for generalized cross-generator detection.
- **Dynamic Attention Weighting:** Automatically adjusts branch reliance per image depending on which domain displays stronger forensic signals.
- **Leak-Free Decision Boundary:** The classification threshold is selected strictly on the validation set using Youden's J statistic ($0.80$), preserving unbiased test evaluation.
- **Interactive Streamlit Web App:** Real-time image upload, probability gauge, confidence score, and live forensic feature map visualizations (FFT spectrum & Color perturbation map).
- **CLI Inference Support:** Fast command-line inference for single images or bulk image directories.
- **Streamlit Community Cloud Ready:** Minimal CPU-compatible dependencies with automatic model artifact resolution.

---

## Reported Benchmark Results

*Reported from the original research and training experiment on a held-out test split (2,218 images: 1,218 Real, 1,000 AI) using a validation-selected threshold of 0.80:*

| Metric | Real Images | AI Images | Overall |
|---|---|---|---|
| **Precision** | 0.93 | 0.95 | **0.94** |
| **Recall** | 0.96 | 0.91 | **0.94** |
| **F1-Score** | 0.94 | 0.93 | **0.94** |

- **Test Accuracy:** `93.78%` (2,080 / 2,218 correct)
- **AUC-ROC:** `0.9748`
- **Optimal Decision Threshold (Validation-Selected):** `0.80`

---

## Project Structure

```text
Real_Fake_image_detetctor/
├── app.py                      # Streamlit interactive web application
├── README.md                   # Project documentation
├── LICENSE                     # MIT License
├── requirements.txt            # Lightweight deployment dependencies (CPU/Streamlit Cloud)
├── requirements-training.txt   # Training & GPU dependencies
├── .gitignore                  # Git ignore configuration
│
├── configs/
│   └── config.py               # Hyperparameters, image dimensions, and threshold config
│
├── src/
│   ├── model.py                # FDCS-Net V4 hybrid neural network architecture
│   ├── data_preprocessing.py   # Image loading, validation, and unified preprocessing
│   ├── predict.py              # CLI inference engine
│   ├── train.py                # 3-stage training pipeline (staged backbone freezing)
│   └── utils.py                # Model compilation, evaluation, loading, and explainability
│
├── models/
│   └── README.md               # Model weights documentation and resolution guidelines
│
├── samples/
│   ├── sample_ai.jpg           # Sample AI-generated test image
│   └── sample_real.jpg         # Sample authentic photography test image
│
├── notebooks/
│   └── fdcsnet-v4-training.ipynb  # End-to-end training notebook (Google Colab)
│
└── reports/
    └── fdcsnet-v4 report.pdf   # Architectural research report
```

---

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/<your-username>/SynthLens.git
cd SynthLens
```

### 2. Set Up a Virtual Environment

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux / macOS
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

For web app inference and deployment:
```bash
pip install -r requirements.txt
```

For model training with GPU support:
```bash
pip install -r requirements-training.txt
```

---

## Model Weights

The trained FDCS-Net V4 model weights file (`models/fdcsnet_v4_final.keras`, ~18.9 MB) is included directly in this repository. 

- **No external downloads required:** The Streamlit app and CLI scripts load `models/fdcsnet_v4_final.keras` directly out-of-the-box.
- **Optional Remote Fallback:** If hosting weights separately or deploying via CI/CD, specifying the `FDCSNET_MODEL_URL` environment variable will allow runtime downloading if the local file is ever missing.

---

## Running the Streamlit Web Application

Launch the local Streamlit dashboard:

```bash
streamlit run app.py
```

The web application opens at `http://localhost:8501`.

### Web App Capabilities:
- **File Upload:** Upload any `.jpg`, `.jpeg`, `.png`, `.webp`, or `.bmp` file (up to 15 MB).
- **Preset Demonstration:** Select built-in sample images to test the model immediately.
- **Classification Output:** Real-time prediction banner (Real vs AI-Generated), probability spectrum $P(\text{AI})$, and confidence percentage.
- **Forensic Feature Inspection:** Displays the intermediate Color Stability Perturbation map and 2-D FFT Magnitude spectrum computed live by FDCS-Net V4.
- **Attention Weights Breakdown:** Visualizes the dynamic attention weighting across Spatial, Color, and Frequency branches.

---

## CLI Inference

Run inference from the terminal using `src/predict.py`:

```bash
# Single image
python src/predict.py --image samples/sample_ai.jpg

# Directory of images
python src/predict.py --dir samples/

# Custom threshold override
python src/predict.py --image samples/sample_real.jpg --threshold 0.75
```

---

## Training Pipeline (Optional)

To train FDCS-Net V4 on your own dataset:

### 1. Organize Dataset
```text
data/
├── train/
│   ├── real/
│   └── ai/
└── test/
    ├── real/
    └── ai/
```

### 2. Run the 3-Stage Training Pipeline
```bash
python src/train.py --train-dir data/train --output-dir models/
```

- **Stage 1 (12 epochs):** Train EfficientNet-B0 spatial branch alone (90% backbone frozen).
- **Stage 2 (10 epochs):** Freeze spatial branch; train Color and Frequency CNN branches with auxiliary supervision heads.
- **Stage 3 (10 epochs):** Unfreeze backbone to 70% and fine-tune all branches end-to-end with attention fusion and auxiliary losses.

---

## Deployment Guide (Streamlit Community Cloud)

1. Push your repository to GitHub.
2. Sign in to [Streamlit Community Cloud](https://share.streamlit.io/).
3. Click **New app**.
4. Select your repository and branch.
5. Set **Main file path** to: `app.py`.
6. (Optional) In **Advanced settings → Secrets**, set:
   ```toml
   FDCSNET_MODEL_URL = "https://huggingface.co/<user>/<repo>/resolve/main/fdcsnet_v4_final.keras"
   ```
7. Click **Deploy!**

---

## Limitations & Disclaimer

- **Probabilistic Nature:** SynthLens outputs a statistical probability based on learned multi-domain forensic patterns. It does not provide deterministic or legally binding proof.
- **Domain Shift:** Synthetic generation models evolve rapidly. Models evaluated primarily on specific generator distributions may experience lower confidence on novel, unseen generative architectures.
- **Post-Processing & Compression:** Heavy social-media compression, extreme resizing, or aggressive re-filtering can attenuate high-frequency FFT artifacts and impact detection confidence.
- **Forensic Scope:** This tool is intended for research, screening, and educational purposes.

---

## Author

**Aman Gupta**

Distributed under the MIT License. See [LICENSE](LICENSE) for details.
---

## Streamlit Community Cloud Deployment

**Use Python 3.11** for this project. The deployment dependencies are pinned to
TensorFlow CPU 2.16.1 and compatible package versions.

If an existing Streamlit deployment is running Python 3.14, delete that app and
redeploy it. In **Advanced settings**, explicitly select **Python 3.11** before
deploying. The main file is `app.py`.

See `STREAMLIT_DEPLOYMENT.md` for the exact deployment steps.

