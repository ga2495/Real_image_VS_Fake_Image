"""
SynthLens — AI-Generated Image Detector
=======================================
Streamlit Web Application powered by the FDCS-Net V4 Deep Learning Architecture.

Author: Aman Gupta
Architecture: Hybrid Spatial (EfficientNet-B0), Frequency (FFT Magnitude CNN),
              and Color-Stability CNN with Softmax Attention Fusion.
"""

import io
import os
import sys
from pathlib import Path
from PIL import Image
import numpy as np
import streamlit as st

# Set up paths so src and configs resolve cleanly
ROOT_DIR = Path(__file__).resolve().parent
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from data_preprocessing import validate_image, preprocess_image
from utils import resolve_and_load_model, predict_single_image, extract_intermediate_maps
from configs.config import CLASSIFICATION_THRESHOLD, IMG_SIZE, COLOR_MAP_SIZE


# ---------------------------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="SynthLens — AI-Generated Image Detector",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for polished, portfolio-grade UI
st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.3rem;
        font-weight: 700;
        color: #1E293B;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #64748B;
        margin-bottom: 1.5rem;
    }
    .pred-card-real {
        background: linear-gradient(135deg, #ECFDF5 0%, #D1FAE5 100%);
        border: 2px solid #10B981;
        border-radius: 12px;
        padding: 20px;
        margin-top: 15px;
        margin-bottom: 20px;
    }
    .pred-card-ai {
        background: linear-gradient(135deg, #FEF2F2 0%, #FEE2E2 100%);
        border: 2px solid #EF4444;
        border-radius: 12px;
        padding: 20px;
        margin-top: 15px;
        margin-bottom: 20px;
    }
    .badge-real {
        background-color: #059669;
        color: white;
        padding: 6px 14px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 1.3rem;
        display: inline-block;
    }
    .badge-ai {
        background-color: #DC2626;
        color: white;
        padding: 6px 14px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 1.3rem;
        display: inline-block;
    }
    .stat-box {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 12px;
        text-align: center;
    }
    .stat-label {
        font-size: 0.8rem;
        color: #64748B;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        font-weight: 600;
    }
    .stat-value {
        font-size: 1.4rem;
        font-weight: 700;
        color: #0F172A;
        margin-top: 4px;
    }
    .disclaimer-box {
        background-color: #FFFBEB;
        border-left: 4px solid #F59E0B;
        padding: 12px 16px;
        border-radius: 4px;
        font-size: 0.88rem;
        color: #92400E;
        margin-top: 25px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Model Caching
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading FDCS-Net V4 neural network...")
def get_cached_model():
    """Load and cache the FDCS-Net V4 model in memory."""
    default_model_path = ROOT_DIR / "models" / "fdcsnet_v4_final.keras"
    return resolve_and_load_model(str(default_model_path), auto_download=True)


# ---------------------------------------------------------------------------
# Sidebar UI
# ---------------------------------------------------------------------------
with st.sidebar:
    st.title("🔍 SynthLens")
    st.caption("AI-Generated vs Real Image Detector")
    st.markdown("---")

    st.subheader("Model Information")
    st.markdown(
        """
        - **Architecture:** FDCS-Net V4 (Hybrid)
        - **Spatial Backbone:** EfficientNet-B0 (ImageNet)
        - **Color Branch:** Perturbation Stability CNN
        - **Frequency Branch:** 2-D FFT Log-Magnitude CNN
        - **Fusion:** Softmax Attention Layer
        - **Decision Threshold:** `0.80` (Validation-selected)
        """
    )
    st.markdown("---")

    st.subheader("Reported Benchmark (Test Set)")
    st.markdown(
        """
        *Reported from original training experiment:*
        - **Test Accuracy:** `93.78%`
        - **AUC-ROC:** `0.9748`
        - **Macro F1-Score:** `0.94`
        - **Optimal Threshold:** `0.80`
        """
    )
    st.markdown("---")

    st.subheader("Supported Image Formats")
    st.write("`JPG`, `JPEG`, `PNG`, `WEBP`, `BMP` (Max 15MB)")
    st.markdown("---")

    st.caption("Author: **Aman Gupta**")


# ---------------------------------------------------------------------------
# Main Content UI
# ---------------------------------------------------------------------------
st.markdown('<div class="main-header">SynthLens — AI-Generated Image Detector</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">FDCS-Net V4 based real-vs-AI image classification and forensic feature inspection</div>', unsafe_allow_html=True)

# Attempt to load model
model = None
model_error = None
try:
    model = get_cached_model()
except Exception as e:
    model_error = str(e)

if model_error:
    st.error(
        f"**Model Load Error:**\n\n{model_error}\n\n"
        "Please ensure `models/fdcsnet_v4_final.keras` is present in the repository."
    )

# Input Section: Tabs for Custom Upload vs Pre-loaded Samples
tab_upload, tab_sample = st.tabs(["📁 Upload Image", "🖼️ Sample Images"])

image_to_analyze = None
image_name = ""

with tab_upload:
    uploaded_file = st.file_uploader(
        "Choose an image to verify...",
        type=["jpg", "jpeg", "png", "webp", "bmp"],
        help="Upload an image up to 15MB for AI detection analysis",
    )
    if uploaded_file is not None:
        is_valid, err_msg = validate_image(uploaded_file)
        if not is_valid:
            st.error(f"Invalid image: {err_msg}")
        else:
            image_to_analyze = uploaded_file
            image_name = uploaded_file.name

with tab_sample:
    sample_dir = ROOT_DIR / "samples"
    sample_options = []
    if sample_dir.exists():
        sample_options = [f.name for f in sample_dir.glob("*.*") if f.suffix.lower() in [".jpg", ".jpeg", ".png", ".webp", ".bmp"]]

    if sample_options:
        selected_sample = st.selectbox("Select a demonstration sample image:", sample_options)
        if selected_sample:
            sample_path = sample_dir / selected_sample
            if sample_path.exists():
                image_to_analyze = sample_path
                image_name = selected_sample
    else:
        st.info("No sample images found in the `samples/` directory.")

# Analysis and Display Section
if image_to_analyze is not None:
    st.markdown("---")
    col_preview, col_controls = st.columns([1, 1], gap="large")

    try:
        if isinstance(image_to_analyze, Path):
            display_img = Image.open(image_to_analyze)
        else:
            image_to_analyze.seek(0)
            display_img = Image.open(io.BytesIO(image_to_analyze.read()))
            image_to_analyze.seek(0)
    except Exception as e:
        st.error(f"Failed to load image preview: {e}")
        display_img = None

    with col_preview:
        st.subheader("Input Image Preview")
        if display_img:
            st.image(display_img, use_container_width=True, caption=f"File: {image_name} ({display_img.width}x{display_img.height} px)")

    with col_controls:
        st.subheader("Forensic Analysis")
        st.markdown(
            "Click **Analyze Image** to run the FDCS-Net V4 multi-branch deep learning model."
        )

        threshold = st.slider(
            "Classification Decision Threshold:",
            min_value=0.10,
            max_value=0.95,
            value=float(CLASSIFICATION_THRESHOLD),
            step=0.01,
            help="Default is 0.80 (selected on the validation set during research to prevent data leakage). Predictions with P(AI) >= Threshold are classified as AI-Generated.",
        )

        analyze_button = st.button("🚀 Analyze Image", type="primary", use_container_width=True, disabled=(model is None))

    if analyze_button and model is not None:
        with st.spinner("Processing spatial, color-stability, and frequency branches..."):
            try:
                # 1. Run prediction
                pred_result = predict_single_image(model, image_to_analyze, threshold=threshold)
                prob_ai = pred_result["prediction"]
                conf = pred_result["confidence"]
                label = pred_result["label"]
                attn = pred_result.get("attention_weights")

                # 2. Extract intermediate maps
                maps = extract_intermediate_maps(image_to_analyze)

                st.markdown("---")
                st.subheader("Prediction Results")

                # Result Card Banner
                if label == "AI-Generated":
                    st.markdown(
                        f"""
                        <div class="pred-card-ai">
                            <div class="badge-ai">AI-GENERATED</div>
                            <div style="font-size: 1.15rem; font-weight: 600; color: #991B1B; margin-top: 10px;">
                                Predicted Probability: {prob_ai * 100:.2f}% (Threshold: {threshold:.2f})
                            </div>
                            <div style="color: #7F1D1D; margin-top: 4px; font-size: 0.95rem;">
                                The model's predicted synthetic probability exceeds the decision threshold of {threshold:.2f}.
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        f"""
                        <div class="pred-card-real">
                            <div class="badge-real">REAL PHOTOGRAPH</div>
                            <div style="font-size: 1.15rem; font-weight: 600; color: #065F46; margin-top: 10px;">
                                Predicted Probability: {prob_ai * 100:.2f}% (Threshold: {threshold:.2f})
                            </div>
                            <div style="color: #064E3B; margin-top: 4px; font-size: 0.95rem;">
                                The model's predicted synthetic probability is below the decision threshold of {threshold:.2f}.
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                # Metric boxes
                c1, c2, c3, c4 = st.columns(4)
                with c1:
                    st.markdown(
                        f'<div class="stat-box"><div class="stat-label">Prediction</div><div class="stat-value">{label}</div></div>',
                        unsafe_allow_html=True,
                    )
                with c2:
                    st.markdown(
                        f'<div class="stat-box"><div class="stat-label">AI Probability P(AI)</div><div class="stat-value">{prob_ai * 100:.2f}%</div></div>',
                        unsafe_allow_html=True,
                    )
                with c3:
                    st.markdown(
                        f'<div class="stat-box"><div class="stat-label">Confidence</div><div class="stat-value">{conf * 100:.2f}%</div></div>',
                        unsafe_allow_html=True,
                    )
                with c4:
                    st.markdown(
                        f'<div class="stat-box"><div class="stat-label">Active Threshold</div><div class="stat-value">{threshold:.2f}</div></div>',
                        unsafe_allow_html=True,
                    )

                st.markdown("<br>", unsafe_allow_html=True)
                st.progress(float(prob_ai), text=f"AI Probability Spectrum: {prob_ai * 100:.1f}% (Threshold: {threshold * 100:.0f}%)")

                # Visual Explainability & Intermediate Feature Maps
                st.markdown("---")
                st.subheader("🔬 Forensic Feature Inspection")
                st.markdown(
                    "FDCS-Net V4 extracts multi-domain forensic cues from the image. Below are the actual intermediate "
                    "representations computed and processed by the respective branches."
                )

                m1, m2, m3 = st.columns(3)
                with m1:
                    st.image(display_img.resize((IMG_SIZE, IMG_SIZE)), use_container_width=True, caption=f"1. Spatial Input ({IMG_SIZE}x{IMG_SIZE})")
                    st.caption("Processed by ImageNet pretrained EfficientNet-B0 backbone.")
                with m2:
                    st.image(maps["color_stability"], use_container_width=True, caption=f"2. Color Stability Map ({COLOR_MAP_SIZE}x{COLOR_MAP_SIZE})")
                    st.caption("Perturbation difference map highlighting pixel-grid quantization instability.")
                with m3:
                    st.image(maps["fft_magnitude"], use_container_width=True, caption=f"3. 2D-FFT Magnitude Spectrum ({COLOR_MAP_SIZE}x{COLOR_MAP_SIZE})")
                    st.caption("Log-scaled 2-D Fourier spectrum revealing high-frequency generative grid artifacts.")

                # Attention Fusion Weights (if available)
                if attn:
                    st.markdown("#### Dynamic Attention Fusion Weights")
                    st.markdown("The attention layer dynamically weights the contribution of each branch for this specific image:")
                    aw_s, aw_c, aw_f = st.columns(3)
                    with aw_s:
                        st.metric("Spatial Branch Weight", f"{attn['spatial'] * 100:.1f}%")
                    with aw_c:
                        st.metric("Color Stability Weight", f"{attn['color'] * 100:.1f}%")
                    with aw_f:
                        st.metric("Frequency Branch Weight", f"{attn['frequency'] * 100:.1f}%")

            except Exception as e:
                st.error(f"Inference error occurred: {str(e)}")

# Technical Details Section
st.markdown("---")
with st.expander("ℹ️ Technical Details & Methodology"):
    st.markdown(
        """
        ### Architecture Overview
        FDCS-Net V4 is a hybrid neural network designed to identify synthetic imagery across disparate generative architectures:
        1. **Spatial Branch:** Uses an **EfficientNet-B0** backbone to capture spatial semantics and unnatural texture anomalies.
        2. **Color Stability Branch:** Evaluates high-frequency perturbation resistance via depthwise convolutions on quantized color channels.
        3. **Frequency Branch:** Employs 2-D Fast Fourier Transform (FFT) log-magnitude mapping to capture periodic generative artifacts in frequency space.
        4. **Attention-Based Fusion:** Instead of simple concatenation, branch feature vectors are L2-normalized and combined via a learned softmax attention weighting mechanism.
        
        ### Threshold Selection Methodology
        The optimal classification threshold (`0.80`) was selected strictly on the **validation set** using Youden's J statistic ($J = \\text{TPR} - \\text{FPR}$) to prevent data leakage onto test evaluations.
        """
    )

# Forensic Disclaimer
st.markdown(
    """
    <div class="disclaimer-box">
        <strong>⚠️ Forensic & Legal Disclaimer:</strong><br>
        SynthLens provides probabilistic machine-learning classifications based on learned statistical and forensic artifacts. 
        It is intended for screening, research, and educational purposes and should <strong>not</strong> be treated as definitive legal or forensic evidence.
    </div>
    """,
    unsafe_allow_html=True,
)
