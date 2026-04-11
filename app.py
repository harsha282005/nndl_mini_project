"""
app.py — Streamlit UI Demo for Fashion-MNIST Product Discovery

Launch with:
    streamlit run app.py

Features:
  • Upload any 28×28 grayscale image OR pick a random test image
  • Run ANN / CNN / Both for side-by-side comparison
  • Display confidence bar chart and top-3 predictions
  • Dark, premium aesthetic matching the project theme
"""

import os
import sys
import numpy as np
import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tensorflow as tf
from PIL import Image

# ── Make sure project root is importable ─────────────────────────────────────
ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.config import CLASS_NAMES, MODELS_DIR, IMG_HEIGHT, IMG_WIDTH

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Fashion-MNIST Product Discovery",
    page_icon="👗",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    .main { background-color: #0f0f1a; }
    .stApp { background: linear-gradient(135deg, #0f0f1a 0%, #1a1a2e 50%, #16213e 100%); }

    .hero-title {
        font-size: 2.8rem;
        font-weight: 700;
        background: linear-gradient(90deg, #00d2ff, #a855f7, #ff6b6b);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        padding: 1rem 0 0.2rem;
    }
    .hero-sub {
        text-align: center;
        color: #8892a4;
        font-size: 1rem;
        margin-bottom: 2rem;
    }
    .result-card {
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 16px;
        padding: 1.4rem;
        margin-bottom: 1rem;
        backdrop-filter: blur(10px);
    }
    .prediction-label {
        font-size: 1.6rem;
        font-weight: 700;
        color: #00d2ff;
    }
    .confidence-text {
        font-size: 1rem;
        color: #a855f7;
        font-weight: 600;
    }
    .metric-box {
        background: rgba(0,210,255,0.08);
        border: 1px solid rgba(0,210,255,0.2);
        border-radius: 12px;
        padding: 1rem;
        text-align: center;
    }
    .stButton > button {
        background: linear-gradient(90deg, #00d2ff, #a855f7);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.6rem 2rem;
        font-weight: 600;
        font-size: 1rem;
        transition: opacity 0.2s;
        width: 100%;
    }
    .stButton > button:hover { opacity: 0.85; }
    .sidebar .stSelectbox label { color: #8892a4 !important; }
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════════

@st.cache_resource(show_spinner="Loading models…")
def load_models():
    """Load saved .keras models; return None if not found."""
    ann_path = os.path.join(MODELS_DIR, "ANN_final.keras")
    cnn_path = os.path.join(MODELS_DIR, "CNN_final.keras")
    ann = tf.keras.models.load_model(ann_path) if os.path.exists(ann_path) else None
    cnn = tf.keras.models.load_model(cnn_path) if os.path.exists(cnn_path) else None
    return ann, cnn


@st.cache_data(show_spinner="Loading Fashion-MNIST test set…")
def load_test_data():
    (_, _), (X_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
    X_test = X_test.astype("float32") / 255.0
    return X_test, y_test


def preprocess_image(img_array):
    """Normalise and reshape to (1, 28, 28) and (1, 28, 28, 1)."""
    img = img_array.astype("float32") / 255.0
    flat  = img.reshape(1, -1)          # ANN format
    cnn4d = img.reshape(1, 28, 28, 1)  # CNN format
    return flat, cnn4d


def predict(model, data, top_k=3):
    probs = model.predict(data, verbose=0)[0]
    top_idx = np.argsort(probs)[::-1][:top_k]
    return probs, top_idx


def confidence_chart(probs, model_name, color):
    """Return a Matplotlib figure with a horizontal bar chart of class probabilities."""
    fig, ax = plt.subplots(figsize=(6, 3.5))
    fig.patch.set_facecolor("#1a1a2e")
    ax.set_facecolor("#16213e")

    sorted_idx = np.argsort(probs)
    sorted_probs = probs[sorted_idx]
    sorted_names = [CLASS_NAMES[i] for i in sorted_idx]

    bar_colors = [color if i == sorted_idx[-1] else "#334155"
                  for i in range(len(CLASS_NAMES))]

    bars = ax.barh(sorted_names, sorted_probs * 100,
                   color=bar_colors, edgecolor="#222", linewidth=0.5)

    ax.set_xlabel("Confidence (%)", color="white", fontsize=9)
    ax.set_title(f"{model_name} — Class Probabilities",
                 color="white", fontsize=10, fontweight="bold")
    ax.tick_params(colors="white", labelsize=8)
    ax.set_xlim(0, 100)
    ax.spines[:].set_color("#333")

    for bar, val in zip(bars, sorted_probs):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                f"{val*100:.1f}%", va="center", color="white", fontsize=7.5)

    plt.tight_layout()
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# UI
# ═══════════════════════════════════════════════════════════════════════════════

st.markdown('<div class="hero-title">👗 Fashion-MNIST Product Discovery</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-sub">Classify clothing items using ANN & CNN deep learning models</div>', unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Settings")
    model_choice = st.selectbox(
        "Select Model",
        options=["Both (Compare)", "ANN only", "CNN only"],
        index=0,
    )
    st.markdown("---")
    st.markdown("### 📋 Class Labels")
    for i, name in enumerate(CLASS_NAMES):
        st.markdown(f"`{i}` {name}")
    st.markdown("---")
    st.markdown("**Project:** NNDL Mini Project")
    st.markdown("**Dataset:** Fashion-MNIST (70k images)")

# ── Load models & test data ───────────────────────────────────────────────────
ann_model, cnn_model = load_models()
X_test, y_test = load_test_data()

# ── Model status ──────────────────────────────────────────────────────────────
col_s1, col_s2 = st.columns(2)
with col_s1:
    if ann_model:
        st.success("✅ ANN model loaded")
    else:
        st.warning("⚠️ ANN model not found — run `python main.py` first")
with col_s2:
    if cnn_model:
        st.success("✅ CNN model loaded")
    else:
        st.warning("⚠️ CNN model not found — run `python main.py` first")

st.markdown("---")

# ── Input Section ─────────────────────────────────────────────────────────────
tab_random, tab_upload = st.tabs(["🎲 Random Test Image", "📤 Upload Image"])

img_array = None
true_label = None

with tab_random:
    col_r1, col_r2 = st.columns([1, 3])
    with col_r1:
        if st.button("🎲 Pick Random Image", key="random_btn"):
            st.session_state["rand_idx"] = np.random.randint(0, len(X_test))

    rand_idx = st.session_state.get("rand_idx", 42)
    img_array  = X_test[rand_idx]
    true_label = CLASS_NAMES[y_test[rand_idx]]

    with col_r2:
        st.markdown(f"**Test image #{rand_idx}** — True label: `{true_label}`")
    fig_img, ax_img = plt.subplots(figsize=(2, 2))
    fig_img.patch.set_facecolor("#1a1a2e")
    ax_img.imshow(img_array, cmap="gray")
    ax_img.axis("off")
    ax_img.set_title(true_label, color="white", fontsize=9)
    fig_img.patch.set_alpha(0)
    col_center = st.columns([2, 1, 2])[1]
    with col_center:
        st.pyplot(fig_img, use_container_width=True)
    plt.close(fig_img)

with tab_upload:
    uploaded = st.file_uploader(
        "Upload a 28×28 grayscale PNG/JPG",
        type=["png", "jpg", "jpeg"]
    )
    if uploaded:
        pil_img = Image.open(uploaded).convert("L").resize((IMG_HEIGHT, IMG_WIDTH))
        img_array = np.array(pil_img)
        true_label = "Unknown"
        fig_up, ax_up = plt.subplots(figsize=(2, 2))
        fig_up.patch.set_facecolor("#1a1a2e")
        ax_up.imshow(img_array, cmap="gray")
        ax_up.axis("off")
        col_up = st.columns([2, 1, 2])[1]
        with col_up:
            st.pyplot(fig_up, use_container_width=True)
        plt.close(fig_up)

# ── Prediction ────────────────────────────────────────────────────────────────
st.markdown("---")
if img_array is not None:
    flat, cnn4d = preprocess_image(img_array)

    run_ann = model_choice in ("Both (Compare)", "ANN only") and ann_model
    run_cnn = model_choice in ("Both (Compare)", "CNN only") and cnn_model

    cols = st.columns(2 if (run_ann and run_cnn) else 1)

    def render_result(col, model, data, model_name, color, emoji):
        probs, top_idx = predict(model, data)
        top_class  = CLASS_NAMES[top_idx[0]]
        confidence = probs[top_idx[0]] * 100

        with col:
            st.markdown(f'<div class="result-card">', unsafe_allow_html=True)
            st.markdown(f"### {emoji} {model_name} Prediction")
            st.markdown(f'<div class="prediction-label">{top_class}</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="confidence-text">Confidence: {confidence:.1f}%</div>', unsafe_allow_html=True)

            if true_label and true_label != "Unknown":
                correct = (top_class == true_label)
                badge = "✅ Correct" if correct else "❌ Wrong"
                st.markdown(f"**Result:** {badge} | True: `{true_label}`")

            st.markdown("**Top-3 Predictions:**")
            for rank, idx in enumerate(top_idx):
                st.markdown(f"{rank+1}. `{CLASS_NAMES[idx]}` — {probs[idx]*100:.1f}%")

            st.markdown('</div>', unsafe_allow_html=True)

            chart = confidence_chart(probs, model_name, color)
            st.pyplot(chart, use_container_width=True)
            plt.close(chart)

    if run_ann and run_cnn:
        render_result(cols[0], ann_model, flat,  "ANN", "#00d2ff", "🧠")
        render_result(cols[1], cnn_model, cnn4d, "CNN", "#a855f7", "🔬")
    elif run_ann:
        render_result(cols[0], ann_model, flat,  "ANN", "#00d2ff", "🧠")
    elif run_cnn:
        render_result(cols[0], cnn_model, cnn4d, "CNN", "#a855f7", "🔬")
    else:
        st.info("No models found. Train them first by running: `python main.py`")

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<div style='text-align:center;color:#555;font-size:0.85rem'>"
    "Fashion-MNIST Product Discovery · NNDL Mini Project · B. Sai Subrahmanyam"
    "</div>",
    unsafe_allow_html=True
)
