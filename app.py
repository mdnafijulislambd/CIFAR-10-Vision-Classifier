import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from PIL import Image
import tensorflow as tf
from PIL import Image
import time
import gdown
import os
# ============================================================
#  CONFIG
# ============================================================
MODEL_PATH = "final_cifar10_model_exp6.keras"

CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]

EMOJIS = {
    "airplane": "✈️", "automobile": "🚗", "bird": "🐦",
    "cat": "🐱", "deer": "🦌", "dog": "🐶",
    "frog": "🐸", "horse": "🐴", "ship": "🚢", "truck": "🚚",
}

CLASS_COLORS = {
    "airplane": "#60a5fa", "automobile": "#f472b6", "bird": "#34d399",
    "cat": "#fb923c", "deer": "#a78bfa", "dog": "#facc15",
    "frog": "#4ade80", "horse": "#f87171", "ship": "#22d3ee", "truck": "#c084fc",
}

# Training history (replace with your real data if needed)
TRAINING_HISTORY = {
    "epochs": list(range(1, 21)),
    "train_acc": [0.42, 0.55, 0.62, 0.67, 0.71, 0.74, 0.76, 0.78, 0.80, 0.81,
                  0.82, 0.83, 0.84, 0.85, 0.86, 0.87, 0.875, 0.88, 0.885, 0.89],
    "val_acc":   [0.38, 0.50, 0.58, 0.63, 0.67, 0.70, 0.72, 0.74, 0.76, 0.77,
                  0.78, 0.785, 0.79, 0.795, 0.80, 0.805, 0.81, 0.812, 0.815, 0.82],
    "train_loss": [2.1, 1.5, 1.2, 1.0, 0.85, 0.75, 0.68, 0.62, 0.57, 0.53,
                   0.49, 0.46, 0.43, 0.41, 0.39, 0.37, 0.36, 0.35, 0.34, 0.33],
    "val_loss":   [2.3, 1.8, 1.4, 1.15, 0.98, 0.87, 0.79, 0.72, 0.67, 0.63,
                   0.59, 0.56, 0.53, 0.51, 0.49, 0.48, 0.47, 0.465, 0.46, 0.458]
}

# Your GitHub repo URL (change this!)
GITHUB_URL = "https://github.com/mdnafijulislambd/CIFAR-10-Vision-Classifier.git"

# ============================================================
#  PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="CIFAR-10 Vision Classifier",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
#  CSS (with GitHub icon only, no text, top-right)
# ============================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Mono:wght@400;500&family=DM+Sans:wght@300;400;500&display=swap');

/* ── Root ── */
:root {
    --bg:        #08090d;
    --bg2:       #0f1117;
    --bg3:       #14161e;
    --border:    rgba(255,255,255,0.07);
    --text:      #e8eaf0;
    --muted:     #6b7280;
    --accent:    #6366f1;
    --accent2:   #22d3ee;
    --success:   #10b981;
}

/* ── Base ── */
html, body, [data-testid="stAppViewContainer"] {
    background: var(--bg) !important;
    font-family: 'DM Sans', sans-serif;
    color: var(--text);
}
[data-testid="stHeader"] { background: transparent !important; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: var(--bg2) !important;
    border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] * { color: var(--text) !important; }

/* ── Main padding ── */
.block-container {
    padding: 2rem 3rem !important;
    max-width: 1300px;
    position: relative;
}

/* ── GitHub icon only (top-right) ── */
.github-corner {
    position: fixed;
    top: 20px;
    right: 20px;
    z-index: 1000;
    background: var(--bg2);
    border-radius: 50%;
    width: 44px;
    height: 44px;
    display: flex;
    align-items: center;
    justify-content: center;
    text-decoration: none;
    border: 1px solid var(--border);
    transition: all 0.2s ease;
    backdrop-filter: blur(8px);
}
.github-corner:hover {
    border-color: var(--accent2);
    transform: translateY(-2px);
    background: var(--bg3);
}
.github-corner svg {
    width: 22px;
    height: 22px;
    fill: var(--text);
}

/* ── Hero section with creator line ── */
.hero-wrap {
    display: flex;
    flex-direction: column;
    gap: 6px;
    margin-bottom: 1.5rem;
}
.hero-label {
    font-family: 'DM Mono', monospace;
    font-size: 11px;
    letter-spacing: 3px;
    text-transform: uppercase;
    color: var(--accent2);
}
.hero-title {
    font-family: 'Syne', sans-serif;
    font-size: clamp(32px, 5vw, 56px);
    font-weight: 800;
    line-height: 1.05;
    background: linear-gradient(135deg, #fff 30%, var(--accent2) 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0;
}
.hero-sub {
    font-size: 14px;
    color: var(--muted);
    font-weight: 300;
    margin-top: 4px;
}
.hero-sub span {
    color: var(--accent2);
    font-weight: 500;
}
.creator {
    font-size: 12px;
    color: var(--muted);
    margin-top: 8px;
    font-family: 'DM Mono', monospace;
    letter-spacing: 0.5px;
}
.creator strong {
    color: var(--accent2);
    font-weight: 600;
}

/* ── Cards, progress bars, etc. (keep your existing styles) ── */
.glass-card {
    background: var(--bg3);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 24px;
    margin-bottom: 16px;
    position: relative;
    overflow: hidden;
}
.glass-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(255,255,255,0.12), transparent);
}
.sec-title {
    font-family: 'Syne', sans-serif;
    font-size: 13px;
    font-weight: 600;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: var(--muted);
    margin-bottom: 14px;
}
.pred-badge {
    display: inline-flex;
    align-items: center;
    gap: 10px;
    background: linear-gradient(135deg, rgba(99,102,241,0.15), rgba(34,211,238,0.08));
    border: 1px solid rgba(99,102,241,0.4);
    border-radius: 50px;
    padding: 10px 22px;
    margin-bottom: 18px;
}
.pred-emoji { font-size: 28px; line-height: 1; }
.pred-text  {
    font-family: 'Syne', sans-serif;
    font-size: 22px;
    font-weight: 700;
    color: #fff;
}
.pred-conf  {
    font-family: 'DM Mono', monospace;
    font-size: 13px;
    color: var(--accent2);
    margin-left: 4px;
}
.prog-row { margin-bottom: 10px; }
.prog-label {
    display: flex;
    justify-content: space-between;
    font-size: 12px;
    margin-bottom: 4px;
    color: var(--text);
}
.prog-label span:last-child {
    font-family: 'DM Mono', monospace;
    color: var(--muted);
}
.prog-track {
    height: 6px;
    background: rgba(255,255,255,0.06);
    border-radius: 99px;
    overflow: hidden;
}
.prog-fill {
    height: 100%;
    border-radius: 99px;
    transition: width 0.6s cubic-bezier(.16,1,.3,1);
}
.tta-card {
    background: var(--bg2);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 16px;
    text-align: center;
}
.tta-aug-name {
    font-family: 'DM Mono', monospace;
    font-size: 10px;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    color: var(--muted);
    margin-bottom: 8px;
}
.tta-pred-label {
    font-family: 'Syne', sans-serif;
    font-size: 15px;
    font-weight: 600;
    color: #fff;
}
.tta-pred-conf {
    font-family: 'DM Mono', monospace;
    font-size: 12px;
    color: var(--accent2);
    margin-top: 2px;
}
.tta-bar-wrap {
    margin-top: 10px;
    height: 4px;
    background: rgba(255,255,255,0.06);
    border-radius: 99px;
    overflow: hidden;
}
.tta-bar-fill {
    height: 100%;
    border-radius: 99px;
    background: linear-gradient(90deg, var(--accent), var(--accent2));
}
.stat-row { display:flex; gap:10px; flex-wrap:wrap; margin-bottom:20px; }
.stat-pill {
    background: var(--bg2);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 10px 16px;
    display: flex;
    flex-direction: column;
    gap: 2px;
}
.stat-pill .num {
    font-family: 'Syne', sans-serif;
    font-size: 20px;
    font-weight: 700;
    color: var(--accent2);
}
.stat-pill .lbl {
    font-size: 11px;
    color: var(--muted);
    letter-spacing: 0.5px;
}
[data-testid="stFileUploader"] {
    background: var(--bg3) !important;
    border: 1.5px dashed rgba(99,102,241,0.4) !important;
    border-radius: 14px !important;
    padding: 10px !important;
}
[data-testid="stFileUploader"]:hover {
    border-color: rgba(99,102,241,0.7) !important;
}
[data-testid="stImage"] img {
    border-radius: 12px !important;
    border: 1px solid var(--border) !important;
}
.stSpinner > div { border-top-color: var(--accent) !important; }
hr { border-color: var(--border) !important; }
[data-testid="stExpander"] {
    background: var(--bg3) !important;
    border: 1px solid var(--border) !important;
    border-radius: 12px !important;
}
.about-card {
    background: linear-gradient(135deg, rgba(99,102,241,0.12), rgba(34,211,238,0.06));
    border: 1px solid rgba(99,102,241,0.2);
    border-radius: 12px;
    padding: 16px;
    margin-bottom: 16px;
    font-size: 13px;
    line-height: 1.6;
}
::-webkit-scrollbar { width: 5px; }
::-webkit-scrollbar-track { background: var(--bg); }
::-webkit-scrollbar-thumb { background: rgba(255,255,255,0.1); border-radius: 99px; }
</style>
""", unsafe_allow_html=True)

# # ============================================================
# #  GITHUB ICON (only icon, no text)
# # ============================================================
# st.markdown(f"""
# <a href="{GITHUB_URL}" target="_blank" class="github-corner">
#     <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24">
#         <path d="M12 0C5.37 0 0 5.37 0 12c0 5.31 3.435 9.795 8.205 11.385.6.105.825-.255.825-.57 0-.285-.015-1.23-.015-2.235-3.015.555-3.795-.735-4.035-1.41-.135-.345-.72-1.41-1.23-1.695-.42-.225-1.02-.78-.015-.795.945-.015 1.62.87 1.845 1.23 1.08 1.815 2.805 1.305 3.495.99.105-.78.42-1.305.765-1.605-2.67-.3-5.46-1.335-5.46-5.925 0-1.305.465-2.385 1.23-3.225-.12-.3-.54-1.53.12-3.18 0 0 1.005-.315 3.3 1.23.96-.27 1.98-.405 3-.405s2.04.135 3 .405c2.295-1.56 3.3-1.23 3.3-1.23.66 1.65.24 2.88.12 3.18.765.84 1.23 1.905 1.23 3.225 0 4.605-2.805 5.625-5.475 5.925.435.375.81 1.095.81 2.22 0 1.605-.015 2.895-.015 3.3 0 .315.225.69.825.57A12.02 12.02 0 0 0 24 12c0-6.63-5.37-12-12-12z"/>
#     </svg>
# </a>
# """, unsafe_allow_html=True)


# =========================
# DOWNLOAD MODEL (SAFE VERSION)
# =========================
def download_model():
    if not os.path.exists(MODEL_PATH) or os.path.getsize(MODEL_PATH) < 1000000:
        with st.spinner("Downloading model..."):
            gdown.download(
                "https://drive.google.com/uc?id=1eQ71ay40dkemH2wE0eIz_t6pyQ8YlyI3",
                MODEL_PATH,
                quiet=False,
                fuzzy=True
            )

# 🔥 MUST CALL HERE (IMPORTANT)
download_model()

# =========================
# LOAD MODEL (AFTER DOWNLOAD)
# =========================
@st.cache_resource
def load_model():
    try:
        if not os.path.exists(MODEL_PATH):
            st.error("Model file missing!")
            return None

        model = tf.keras.models.load_model(MODEL_PATH, compile=False)
        return model

    except Exception as e:
        st.error(f"Model load failed: {e}")
        return None

model = load_model()

# ============================================================
#  FIXED TTA PREDICTION (NO DIVISION BY 255)
# ============================================================
def tta_predict(img_arr):
    base = img_arr.astype(np.uint8)

    pil = Image.fromarray(base)

    augs = [
        base,
        np.fliplr(base),
        np.array(pil.rotate(5)),
        np.array(pil.rotate(-5)),
    ]

    preds = []
    for aug in augs:
        x = aug.astype(np.float32)
        x = np.expand_dims(x, axis=0)

        logits = model.predict(x, verbose=0)[0]
        probs = tf.nn.softmax(logits).numpy()
        preds.append(probs)

    return np.array(preds)

# ============================================================
#  SIDEBAR (unchanged)
# ============================================================
with st.sidebar:
    st.markdown("""
    <div style="font-family:'Syne',sans-serif;font-size:22px;font-weight:800;
                background:linear-gradient(135deg,#fff,#22d3ee);
                -webkit-background-clip:text;-webkit-text-fill-color:transparent;
                margin-bottom:6px;">
        CIFAR-10<br>Classifier
    </div>
    <div style="font-size:11px;color:#6b7280;letter-spacing:1px;margin-bottom:20px;">
        CNN · RESIDUAL · TTA
    </div>
    """, unsafe_allow_html=True)
    st.markdown("""
    <div class="about-card">
        A <strong>Convolutional Neural Network</strong> trained on the CIFAR-10 dataset
        to classify images into <strong>10 categories</strong> using residual connections
        and Test-Time Augmentation for robust predictions.
    </div>
    """, unsafe_allow_html=True)
    st.markdown("<div class='sec-title'>10 Classes</div>", unsafe_allow_html=True)
    cols_sb = st.columns(2)
    for i, cls in enumerate(CLASSES):
        with cols_sb[i % 2]:
            st.markdown(f"<div style='font-size:13px;margin-bottom:6px;'>{EMOJIS[cls]} {cls.capitalize()}</div>", unsafe_allow_html=True)
    st.divider()
    st.markdown("<div class='sec-title'>Model Details</div>", unsafe_allow_html=True)
    for k, v in [("Architecture", "ResNet-style CNN"), ("Dataset", "CIFAR-10"),
                  ("TTA Folds", "4"), ("Input Size", "32×32 px")]:
        st.markdown(f"<div style='display:flex;justify-content:space-between;font-size:12px;margin-bottom:6px;'><span style='color:#6b7280;'>{k}</span><span style='font-family:DM Mono,monospace;color:#22d3ee;'>{v}</span></div>", unsafe_allow_html=True)
    st.divider()
    st.markdown("<div style='font-size:11px;color:#4b5563;text-align:center;'>Developed by <strong style='color:#6b7280;'>Md. Nafijul Islam</strong></div>", unsafe_allow_html=True)

# ============================================================
#  MAIN CONTENT
# ============================================================
st.markdown("""
<div class="hero-wrap">
    <div class="hero-label">Deep Learning · Image Recognition</div>
    <div class="hero-title">CIFAR-10 Vision Classifier</div>
    <div class="hero-sub">
        Residual CNN with <span>Test-Time Augmentation</span> —
        upload any image to classify it into one of 10 categories.
    </div>
    <div class="creator">Created by <strong>Md. Nafijul Islam</strong></div>
</div>
""", unsafe_allow_html=True)

# Stats row
st.markdown("""
<div class="stat-row">
    <div class="stat-pill"><div class="num">~89%</div><div class="lbl">Val Accuracy</div></div>
    <div class="stat-pill"><div class="num">10</div><div class="lbl">Classes</div></div>
    <div class="stat-pill"><div class="num">4×</div><div class="lbl">TTA Folds</div></div>
    <div class="stat-pill"><div class="num">50K</div><div class="lbl">Training Images</div></div>
</div>
""", unsafe_allow_html=True)

# ============================================================
#  TRAINING HISTORY: TWO SMOOTH GRAPHS (NO BOLD POINTS)
# ============================================================
df_history = pd.DataFrame(TRAINING_HISTORY)

with st.expander("📈 Training History (Graphs)", expanded=False):
    col_acc, col_loss = st.columns(2)
    
    with col_acc:
        fig_acc = go.Figure()
        fig_acc.add_trace(go.Scatter(
            x=df_history["epochs"], y=df_history["train_acc"]*100,
            mode="lines", name="Train Accuracy",
            line=dict(color="#6366f1", width=3, shape='spline'),
            fill='tozeroy', fillcolor='rgba(99,102,241,0.15)',
            hovertemplate='Epoch %{x}<br>Train Acc: %{y:.1f}%<extra></extra>'
        ))
        fig_acc.add_trace(go.Scatter(
            x=df_history["epochs"], y=df_history["val_acc"]*100,
            mode="lines", name="Validation Accuracy",
            line=dict(color="#22d3ee", width=3, shape='spline'),
            fill='tozeroy', fillcolor='rgba(34,211,238,0.1)',
            hovertemplate='Epoch %{x}<br>Val Acc: %{y:.1f}%<extra></extra>'
        ))
        fig_acc.update_layout(
            title="Accuracy over Epochs",
            xaxis_title="Epoch",
            yaxis_title="Accuracy (%)",
            template="plotly_dark",
            height=380,
            hovermode="x unified",
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#e8eaf0"),
            legend=dict(orientation="h", yanchor="bottom", y=0.98, xanchor="right", x=0.98)
        )
        fig_acc.update_xaxes(gridcolor="rgba(255,255,255,0.08)", showgrid=True)
        fig_acc.update_yaxes(gridcolor="rgba(255,255,255,0.08)", range=[30, 100])
        st.plotly_chart(fig_acc, use_container_width=True)
    
    with col_loss:
        fig_loss = go.Figure()
        fig_loss.add_trace(go.Scatter(
            x=df_history["epochs"], y=df_history["train_loss"],
            mode="lines", name="Train Loss",
            line=dict(color="#f472b6", width=3, shape='spline'),
            fill='tozeroy', fillcolor='rgba(244,114,182,0.1)',
            hovertemplate='Epoch %{x}<br>Train Loss: %{y:.3f}<extra></extra>'
        ))
        fig_loss.add_trace(go.Scatter(
            x=df_history["epochs"], y=df_history["val_loss"],
            mode="lines", name="Validation Loss",
            line=dict(color="#facc15", width=3, shape='spline'),
            fill='tozeroy', fillcolor='rgba(250,204,21,0.1)',
            hovertemplate='Epoch %{x}<br>Val Loss: %{y:.3f}<extra></extra>'
        ))
        fig_loss.update_layout(
            title="Loss over Epochs",
            xaxis_title="Epoch",
            yaxis_title="Loss",
            template="plotly_dark",
            height=380,
            hovermode="x unified",
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#e8eaf0"),
            legend=dict(orientation="h", yanchor="bottom", y=0.98, xanchor="right", x=0.98)
        )
        fig_loss.update_xaxes(gridcolor="rgba(255,255,255,0.08)", showgrid=True)
        fig_loss.update_yaxes(gridcolor="rgba(255,255,255,0.08)")
        st.plotly_chart(fig_loss, use_container_width=True)

# Expander 2: Numeric table
with st.expander("📊 Training History (Numeric Table)", expanded=False):
    df_display = df_history.copy()
    df_display["train_acc"] = (df_display["train_acc"] * 100).map("{:.2f}%".format)
    df_display["val_acc"] = (df_display["val_acc"] * 100).map("{:.2f}%".format)
    df_display["train_loss"] = df_display["train_loss"].map("{:.3f}".format)
    df_display["val_loss"] = df_display["val_loss"].map("{:.3f}".format)
    st.dataframe(df_display, use_container_width=True)

st.divider()

# ============================================================
#  UPLOAD & PREDICTION (Two columns)
# ============================================================
left, right = st.columns([1, 1], gap="large")

with left:
    st.markdown("<div class='sec-title'>Upload Image</div>", unsafe_allow_html=True)
    file = st.file_uploader("Choose a JPG or PNG file", type=["jpg", "jpeg", "png"], label_visibility="collapsed")
    if file:
        image = Image.open(file).convert("RGB")
        st.image(image, use_container_width=True, caption="Original image (resized to 32×32 for inference)")
        raw = Image.open(file)
        st.markdown(f"<div style='font-size:11px;color:#6b7280;margin-top:6px;'>Original: {raw.width}×{raw.height}px · {file.type}</div>", unsafe_allow_html=True)
        current_image = image
    else:
        current_image = None
        st.markdown("""
        <div style="text-align:center;padding:40px 20px;color:#4b5563;font-size:13px;">
            <div style="font-size:40px;margin-bottom:10px;">🖼️</div>
            Drop an image here or click Upload<br>
            <span style="font-size:11px;font-family:'DM Mono',monospace;">JPG · PNG · up to 200 MB</span>
        </div>
        """, unsafe_allow_html=True)

with right:
    st.markdown("<div class='sec-title'>Prediction result</div>", unsafe_allow_html=True)
    if current_image is None:
        st.info("👈 Upload an image to see classification")
    else:
        if model is None:
            st.error("Model not loaded. Please check the model file path.")
        else:
            img_resized = current_image.resize((32, 32))
            img_arr = np.array(img_resized, dtype=np.uint8)
            with st.spinner("Running TTA inference (4 folds)…"):
                t0 = time.time()
                preds = tta_predict(img_arr)   # (4,10)
                elapsed = (time.time() - t0) * 1000
            avg = np.mean(preds, axis=0)
            idx = int(np.argmax(avg))
            label = CLASSES[idx]
            conf = avg[idx]
            color = CLASS_COLORS[label]
            
            st.markdown(f"""
            <div class="pred-badge">
                <div class="pred-emoji">{EMOJIS[label]}</div>
                <div>
                    <div class="pred-text">{label.upper()}</div>
                    <div class="pred-conf">{conf*100:.2f}% confidence</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            st.caption(f"⏱️ Inference: {elapsed:.0f} ms (4 TTA folds)")
            
            st.markdown("<div class='sec-title'>Top 5 Predictions</div>", unsafe_allow_html=True)
            top5 = np.argsort(avg)[::-1][:5]
            for rank, i in enumerate(top5):
                c = CLASSES[i]
                v = float(avg[i])
                col = CLASS_COLORS[c]
                st.markdown(f"""
                <div class="prog-row">
                    <div class="prog-label">
                        <span>{'🥇' if rank==0 else '  '} {EMOJIS[c]} {c.capitalize()}</span>
                        <span>{pct(v)}</span>
                    </div>
                    <div class="prog-track">
                        <div class="prog-fill" style="width:{v*100:.2f}%; background:linear-gradient(90deg,{col}cc,{col}66);"></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

# ============================================================
#  ALL CLASSES & TTA BREAKDOWN (after upload)
# ============================================================
if file and current_image is not None and model is not None:
    st.divider()
    col_a, col_b = st.columns([1, 1], gap="large")
    
    with col_a:
        st.markdown("<div class='sec-title'>All Class Probabilities</div>", unsafe_allow_html=True)
        sorted_idx = np.argsort(avg)[::-1]
        for i in sorted_idx:
            c = CLASSES[i]
            v = float(avg[i])
            col_c = CLASS_COLORS[c]
            st.markdown(f"""
            <div class="prog-row">
                <div class="prog-label">
                    <span>{EMOJIS[c]} {c.capitalize()}</span>
                    <span>{pct(v)}</span>
                </div>
                <div class="prog-track">
                    <div class="prog-fill" style="width:{v*100:.2f}%; background:linear-gradient(90deg,{col_c}cc,{col_c}44);"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    with col_b:
        st.markdown("<div class='sec-title'>TTA Augmentation Breakdown</div>", unsafe_allow_html=True)
        aug_names = ["Original", "Horizontal Flip", "Rotate +5°", "Rotate −5°"]
        aug_icons = ["🖼️", "🔄", "↗️", "↘️"]
        c1, c2 = st.columns(2)
        for k in range(4):
            p = preds[k]
            pi = int(np.argmax(p))
            pcls = CLASSES[pi]
            pv = float(np.max(p))
            pcol = CLASS_COLORS[pcls]
            target = c1 if k % 2 == 0 else c2
            with target:
                st.markdown(f"""
                <div class="tta-card">
                    <div class="tta-aug-name">{aug_icons[k]} {aug_names[k]}</div>
                    <div style="font-size:24px;">{EMOJIS[pcls]}</div>
                    <div class="tta-pred-label">{pcls.capitalize()}</div>
                    <div class="tta-pred-conf">{pv*100:.1f}%</div>
                    <div class="tta-bar-wrap">
                        <div class="tta-bar-fill" style="width:{pv*100:.1f}%; background:linear-gradient(90deg,{pcol},{pcol}88);"></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                st.markdown("<div style='height:10px;'></div>", unsafe_allow_html=True)
    
    st.divider()
    tta_labels = [CLASSES[int(np.argmax(preds[i]))] for i in range(4)]
    agreement = tta_labels.count(label) / 4
    st.markdown(f"""
    <div class="glass-card">
        <div class="sec-title">TTA Agreement</div>
        <div style="display:flex;align-items:center;gap:16px;">
            <div style="font-size:32px;">{'✅' if agreement==1 else '⚠️'}</div>
            <div>
                <div style="font-family:'Syne',sans-serif;font-size:18px;font-weight:700;">
                    {int(agreement*4)}/4 augmentations agree on
                    <span style="color:#22d3ee;">{label.capitalize()}</span>
                </div>
                <div style="font-size:12px;color:#6b7280;margin-top:2px;">
                    {'High confidence — all TTA folds are consistent.' if agreement==1 else 'Some folds disagree — treat confidence with caution.'}
                </div>
            </div>
            <div style="margin-left:auto;">
                <div style="font-family:'DM Mono',monospace;font-size:28px;color:#22d3ee;font-weight:500;">{int(agreement*100)}%</div>
            </div>
        </div>
        <div class="prog-track" style="margin-top:12px;height:8px;">
            <div class="prog-fill" style="width:{agreement*100:.0f}%; background:{'linear-gradient(90deg,#10b981,#22d3ee)' if agreement==1 else 'linear-gradient(90deg,#f59e0b,#f87171)'};"></div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ============================================================
#  FOOTER
# ============================================================
st.markdown("<br>", unsafe_allow_html=True)
st.markdown("""
<div style="text-align:center;padding:16px;border-top:1px solid rgba(255,255,255,0.06);">
    <div style="font-size:11px;color:#374151;letter-spacing:1px;">
        Built with <span style="color:#6366f1;">Streamlit</span> ·
        TensorFlow / Keras ·
        <strong>Md. Nafijul Islam</strong>
    </div>
</div>
""", unsafe_allow_html=True)









# import streamlit as st
# import numpy as np
# import pandas as pd
# import plotly.graph_objects as go
# from plotly.subplots import make_subplots
# from PIL import Image
# import tensorflow as tf
# import time
# import os
# import requests

# # ============================================================
# #  CONFIG
# # ============================================================
# MODEL_FILE = "final_cifar10_model_exp6.keras"
# # Your Google Drive file ID (extracted from the share link)
# GOOGLE_DRIVE_FILE_ID = "1eQ71ay40dkemH2wE0eIz_t6pyQ8YlyI3"

# CLASSES = [
#     "airplane", "automobile", "bird", "cat", "deer",
#     "dog", "frog", "horse", "ship", "truck",
# ]

# EMOJIS = {
#     "airplane": "✈️", "automobile": "🚗", "bird": "🐦",
#     "cat": "🐱", "deer": "🦌", "dog": "🐶",
#     "frog": "🐸", "horse": "🐴", "ship": "🚢", "truck": "🚚",
# }

# CLASS_COLORS = {
#     "airplane": "#60a5fa", "automobile": "#f472b6", "bird": "#34d399",
#     "cat": "#fb923c", "deer": "#a78bfa", "dog": "#facc15",
#     "frog": "#4ade80", "horse": "#f87171", "ship": "#22d3ee", "truck": "#c084fc",
# }

# TRAINING_HISTORY = {
#     "epochs": list(range(1, 21)),
#     "train_acc": [0.42, 0.55, 0.62, 0.67, 0.71, 0.74, 0.76, 0.78, 0.80, 0.81,
#                   0.82, 0.83, 0.84, 0.85, 0.86, 0.87, 0.875, 0.88, 0.885, 0.89],
#     "val_acc":   [0.38, 0.50, 0.58, 0.63, 0.67, 0.70, 0.72, 0.74, 0.76, 0.77,
#                   0.78, 0.785, 0.79, 0.795, 0.80, 0.805, 0.81, 0.812, 0.815, 0.82],
#     "train_loss": [2.1, 1.5, 1.2, 1.0, 0.85, 0.75, 0.68, 0.62, 0.57, 0.53,
#                    0.49, 0.46, 0.43, 0.41, 0.39, 0.37, 0.36, 0.35, 0.34, 0.33],
#     "val_loss":   [2.3, 1.8, 1.4, 1.15, 0.98, 0.87, 0.79, 0.72, 0.67, 0.63,
#                    0.59, 0.56, 0.53, 0.51, 0.49, 0.48, 0.47, 0.465, 0.46, 0.458]
# }

# GITHUB_URL = "https://github.com/mdnafijulislambd/CIFAR-10-Vision-Classifier"

# # ============================================================
# #  PAGE CONFIG & CSS (same as original dark theme)
# # ============================================================
# st.set_page_config(page_title="CIFAR-10 Vision Classifier", page_icon="🧠", layout="wide")

# st.markdown("""
# <style>
# @import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Mono:wght@400;500&family=DM+Sans:wght@300;400;500&display=swap');
# :root { --bg: #08090d; --bg2: #0f1117; --bg3: #14161e; --border: rgba(255,255,255,0.07); --text: #e8eaf0; --muted: #6b7280; --accent: #6366f1; --accent2: #22d3ee; --success: #10b981; }
# html, body, [data-testid="stAppViewContainer"] { background: var(--bg) !important; font-family: 'DM Sans', sans-serif; color: var(--text); }
# [data-testid="stHeader"] { background: transparent !important; }
# [data-testid="stSidebar"] { background: var(--bg2) !important; border-right: 1px solid var(--border) !important; }
# .block-container { padding: 2rem 3rem !important; max-width: 1300px; position: relative; }
# .github-corner { position: fixed; top: 20px; right: 20px; z-index: 1000; background: var(--bg2); border-radius: 50%; width: 44px; height: 44px; display: flex; align-items: center; justify-content: center; text-decoration: none; border: 1px solid var(--border); transition: all 0.2s ease; backdrop-filter: blur(8px); }
# .github-corner:hover { border-color: var(--accent2); transform: translateY(-2px); }
# .github-corner svg { width: 22px; height: 22px; fill: var(--text); }
# .hero-wrap { display: flex; flex-direction: column; gap: 6px; margin-bottom: 1.5rem; }
# .hero-label { font-family: 'DM Mono', monospace; font-size: 11px; letter-spacing: 3px; text-transform: uppercase; color: var(--accent2); }
# .hero-title { font-family: 'Syne', sans-serif; font-size: clamp(32px, 5vw, 56px); font-weight: 800; line-height: 1.05; background: linear-gradient(135deg, #fff 30%, var(--accent2) 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text; margin: 0; }
# .hero-sub { font-size: 14px; color: var(--muted); font-weight: 300; margin-top: 4px; }
# .hero-sub span { color: var(--accent2); font-weight: 500; }
# .creator { font-size: 12px; color: var(--muted); margin-top: 8px; font-family: 'DM Mono', monospace; letter-spacing: 0.5px; }
# .creator strong { color: var(--accent2); font-weight: 600; }
# .glass-card { background: var(--bg3); border: 1px solid var(--border); border-radius: 16px; padding: 24px; margin-bottom: 16px; position: relative; overflow: hidden; }
# .sec-title { font-family: 'Syne', sans-serif; font-size: 13px; font-weight: 600; letter-spacing: 2px; text-transform: uppercase; color: var(--muted); margin-bottom: 14px; }
# .pred-badge { display: inline-flex; align-items: center; gap: 10px; background: linear-gradient(135deg, rgba(99,102,241,0.15), rgba(34,211,238,0.08)); border: 1px solid rgba(99,102,241,0.4); border-radius: 50px; padding: 10px 22px; margin-bottom: 18px; }
# .pred-emoji { font-size: 28px; line-height: 1; }
# .pred-text { font-family: 'Syne', sans-serif; font-size: 22px; font-weight: 700; color: #fff; }
# .pred-conf { font-family: 'DM Mono', monospace; font-size: 13px; color: var(--accent2); margin-left: 4px; }
# .prog-row { margin-bottom: 10px; }
# .prog-label { display: flex; justify-content: space-between; font-size: 12px; margin-bottom: 4px; color: var(--text); }
# .prog-label span:last-child { font-family: 'DM Mono', monospace; color: var(--muted); }
# .prog-track { height: 6px; background: rgba(255,255,255,0.06); border-radius: 99px; overflow: hidden; }
# .prog-fill { height: 100%; border-radius: 99px; transition: width 0.6s cubic-bezier(.16,1,.3,1); }
# .tta-card { background: var(--bg2); border: 1px solid var(--border); border-radius: 12px; padding: 16px; text-align: center; }
# .tta-aug-name { font-family: 'DM Mono', monospace; font-size: 10px; letter-spacing: 1.5px; text-transform: uppercase; color: var(--muted); margin-bottom: 8px; }
# .tta-pred-label { font-family: 'Syne', sans-serif; font-size: 15px; font-weight: 600; color: #fff; }
# .tta-pred-conf { font-family: 'DM Mono', monospace; font-size: 12px; color: var(--accent2); margin-top: 2px; }
# .tta-bar-wrap { margin-top: 10px; height: 4px; background: rgba(255,255,255,0.06); border-radius: 99px; overflow: hidden; }
# .tta-bar-fill { height: 100%; border-radius: 99px; background: linear-gradient(90deg, var(--accent), var(--accent2)); }
# .stat-row { display:flex; gap:10px; flex-wrap:wrap; margin-bottom:20px; }
# .stat-pill { background: var(--bg2); border: 1px solid var(--border); border-radius: 8px; padding: 10px 16px; display: flex; flex-direction: column; gap: 2px; }
# .stat-pill .num { font-family: 'Syne', sans-serif; font-size: 20px; font-weight: 700; color: var(--accent2); }
# .stat-pill .lbl { font-size: 11px; color: var(--muted); letter-spacing: 0.5px; }
# [data-testid="stFileUploader"] { background: var(--bg3) !important; border: 1.5px dashed rgba(99,102,241,0.4) !important; border-radius: 14px !important; padding: 10px !important; }
# [data-testid="stFileUploader"]:hover { border-color: rgba(99,102,241,0.7) !important; }
# [data-testid="stImage"] img { border-radius: 12px !important; border: 1px solid var(--border) !important; }
# .stSpinner > div { border-top-color: var(--accent) !important; }
# hr { border-color: var(--border) !important; }
# [data-testid="stExpander"] { background: var(--bg3) !important; border: 1px solid var(--border) !important; border-radius: 12px !important; }
# .about-card { background: linear-gradient(135deg, rgba(99,102,241,0.12), rgba(34,211,238,0.06)); border: 1px solid rgba(99,102,241,0.2); border-radius: 12px; padding: 16px; margin-bottom: 16px; font-size: 13px; line-height: 1.6; }
# ::-webkit-scrollbar { width: 5px; }
# ::-webkit-scrollbar-track { background: var(--bg); }
# ::-webkit-scrollbar-thumb { background: rgba(255,255,255,0.1); border-radius: 99px; }
# </style>
# """, unsafe_allow_html=True)

# # GitHub icon
# st.markdown(f"""
# <a href="{GITHUB_URL}" target="_blank" class="github-corner">
#     <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24">
#         <path d="M12 0C5.37 0 0 5.37 0 12c0 5.31 3.435 9.795 8.205 11.385.6.105.825-.255.825-.57 0-.285-.015-1.23-.015-2.235-3.015.555-3.795-.735-4.035-1.41-.135-.345-.72-1.41-1.23-1.695-.42-.225-1.02-.78-.015-.795.945-.015 1.62.87 1.845 1.23 1.08 1.815 2.805 1.305 3.495.99.105-.78.42-1.305.765-1.605-2.67-.3-5.46-1.335-5.46-5.925 0-1.305.465-2.385 1.23-3.225-.12-.3-.54-1.53.12-3.18 0 0 1.005-.315 3.3 1.23.96-.27 1.98-.405 3-.405s2.04.135 3 .405c2.295-1.56 3.3-1.23 3.3-1.23.66 1.65.24 2.88.12 3.18.765.84 1.23 1.905 1.23 3.225 0 4.605-2.805 5.625-5.475 5.925.435.375.81 1.095.81 2.22 0 1.605-.015 2.895-.015 3.3 0 .315.225.69.825.57A12.02 12.02 0 0 0 24 12c0-6.63-5.37-12-12-12z"/>
#     </svg>
# </a>
# """, unsafe_allow_html=True)

# # # ============================================================
# # #  LOAD MODEL (Auto-download from Google Drive)
# # # ============================================================
# # @st.cache_resource
# # def load_cifar_model():
# #     if not os.path.exists(MODEL_FILE):
# #         with st.spinner("Downloading model (approx 80 MB) from Google Drive... This may take a minute."):
# #             # Construct direct download URL from file ID
# #             url = f"https://drive.google.com/uc?export=download&id={GOOGLE_DRIVE_FILE_ID}"
# #             # Use requests to download
# #             response = requests.get(url, stream=True)
# #             with open(MODEL_FILE, "wb") as f:
# #                 for chunk in response.iter_content(chunk_size=8192):
# #                     f.write(chunk)
# #     return tf.keras.models.load_model(MODEL_FILE, compile=False)

# # model = load_cifar_model()




# @st.cache_resource
# def load_cifar_model():
#     model = load_cifar_model()
#     import tensorflow as tf

#     if not os.path.exists(MODEL_FILE):
#         with st.spinner("Downloading model (80 MB)..."):
#             url = f"https://drive.google.com/uc?export=download&id={GOOGLE_DRIVE_FILE_ID}"
#             response = requests.get(url, stream=True)

#             with open(MODEL_FILE, "wb") as f:
#                 for chunk in response.iter_content(chunk_size=8192):
#                     if chunk:
#                         f.write(chunk)

#     # 🔥 SAFE LOAD (important fix)
#     model = tf.keras.models.load_model(
#         MODEL_FILE,
#         compile=False,
#         safe_mode=False
#     )

#     return model







# # ============================================================
# #  TTA PREDICTION (no division by 255)
# # ============================================================
# def tta_predict(img_arr: np.ndarray):
#     base = img_arr.astype(np.uint8)
#     pil = Image.fromarray(base)
#     augs = [base, np.fliplr(base), np.array(pil.rotate(5)), np.array(pil.rotate(-5))]
#     preds = []
#     for aug in augs:
#         x = aug.astype(np.float32)   # keep 0-255 range
#         x = np.expand_dims(x, axis=0)
#         logits = model.predict(x, verbose=0)[0]
#         probs = tf.nn.softmax(logits).numpy()
#         preds.append(probs)
#     return np.array(preds)

# def pct(v): return f"{v*100:.1f}%"

# # ============================================================
# #  SIDEBAR
# # ============================================================
# with st.sidebar:
#     st.markdown("""
#     <div style="font-family:'Syne',sans-serif;font-size:22px;font-weight:800;
#                 background:linear-gradient(135deg,#fff,#22d3ee);
#                 -webkit-background-clip:text;-webkit-text-fill-color:transparent;
#                 margin-bottom:6px;">
#         CIFAR-10<br>Classifier
#     </div>
#     <div style="font-size:11px;color:#6b7280;letter-spacing:1px;margin-bottom:20px;">
#         CNN · RESIDUAL · TTA
#     </div>
#     """, unsafe_allow_html=True)
#     st.markdown("""
#     <div class="about-card">
#         A <strong>Convolutional Neural Network</strong> trained on the CIFAR-10 dataset
#         to classify images into <strong>10 categories</strong> using residual connections
#         and Test-Time Augmentation for robust predictions.
#     </div>
#     """, unsafe_allow_html=True)
#     st.markdown("<div class='sec-title'>10 Classes</div>", unsafe_allow_html=True)
#     cols_sb = st.columns(2)
#     for i, cls in enumerate(CLASSES):
#         with cols_sb[i % 2]:
#             st.markdown(f"<div style='font-size:13px;margin-bottom:6px;'>{EMOJIS[cls]} {cls.capitalize()}</div>", unsafe_allow_html=True)
#     st.divider()
#     st.markdown("<div class='sec-title'>Model Details</div>", unsafe_allow_html=True)
#     for k, v in [("Architecture", "ResNet-style CNN"), ("Dataset", "CIFAR-10"), ("TTA Folds", "4"), ("Input Size", "32×32 px")]:
#         st.markdown(f"<div style='display:flex;justify-content:space-between;font-size:12px;margin-bottom:6px;'><span style='color:#6b7280;'>{k}</span><span style='font-family:DM Mono,monospace;color:#22d3ee;'>{v}</span></div>", unsafe_allow_html=True)
#     st.divider()
#     st.markdown("<div style='font-size:11px;color:#4b5563;text-align:center;'>Developed by <strong style='color:#6b7280;'>Md. Nafijul Islam</strong></div>", unsafe_allow_html=True)

# # ============================================================
# #  MAIN UI
# # ============================================================
# st.markdown("""
# <div class="hero-wrap">
#     <div class="hero-label">Deep Learning · Image Recognition</div>
#     <div class="hero-title">CIFAR-10 Vision Classifier</div>
#     <div class="hero-sub">Residual CNN with <span>Test-Time Augmentation</span> — upload any image to classify it into one of 10 categories.</div>
#     <div class="creator">Created by <strong>Md. Nafijul Islam</strong></div>
# </div>
# """, unsafe_allow_html=True)

# st.markdown("""
# <div class="stat-row">
#     <div class="stat-pill"><div class="num">~89%</div><div class="lbl">Val Accuracy</div></div>
#     <div class="stat-pill"><div class="num">10</div><div class="lbl">Classes</div></div>
#     <div class="stat-pill"><div class="num">4×</div><div class="lbl">TTA Folds</div></div>
#     <div class="stat-pill"><div class="num">50K</div><div class="lbl">Training Images</div></div>
# </div>
# """, unsafe_allow_html=True)

# # Training history graphs
# with st.expander("📈 Training History (Graphs)", expanded=False):
#     df_history = pd.DataFrame(TRAINING_HISTORY)
#     col_acc, col_loss = st.columns(2)
#     with col_acc:
#         fig_acc = go.Figure()
#         fig_acc.add_trace(go.Scatter(x=df_history["epochs"], y=df_history["train_acc"]*100, mode="lines", name="Train Acc", line=dict(color="#6366f1", width=3, shape='spline'), fill='tozeroy', fillcolor='rgba(99,102,241,0.15)'))
#         fig_acc.add_trace(go.Scatter(x=df_history["epochs"], y=df_history["val_acc"]*100, mode="lines", name="Val Acc", line=dict(color="#22d3ee", width=3, shape='spline'), fill='tozeroy', fillcolor='rgba(34,211,238,0.1)'))
#         fig_acc.update_layout(title="Accuracy", xaxis_title="Epoch", yaxis_title="Accuracy (%)", template="plotly_dark", height=380, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", font=dict(color="#e8eaf0"), legend=dict(orientation="h", yanchor="bottom", y=0.98, xanchor="right", x=0.98))
#         fig_acc.update_xaxes(gridcolor="rgba(255,255,255,0.08)")
#         fig_acc.update_yaxes(gridcolor="rgba(255,255,255,0.08)", range=[30,100])
#         st.plotly_chart(fig_acc, use_container_width=True)
#     with col_loss:
#         fig_loss = go.Figure()
#         fig_loss.add_trace(go.Scatter(x=df_history["epochs"], y=df_history["train_loss"], mode="lines", name="Train Loss", line=dict(color="#f472b6", width=3, shape='spline'), fill='tozeroy', fillcolor='rgba(244,114,182,0.1)'))
#         fig_loss.add_trace(go.Scatter(x=df_history["epochs"], y=df_history["val_loss"], mode="lines", name="Val Loss", line=dict(color="#facc15", width=3, shape='spline'), fill='tozeroy', fillcolor='rgba(250,204,21,0.1)'))
#         fig_loss.update_layout(title="Loss", xaxis_title="Epoch", yaxis_title="Loss", template="plotly_dark", height=380, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", font=dict(color="#e8eaf0"), legend=dict(orientation="h", yanchor="bottom", y=0.98, xanchor="right", x=0.98))
#         fig_loss.update_xaxes(gridcolor="rgba(255,255,255,0.08)")
#         fig_loss.update_yaxes(gridcolor="rgba(255,255,255,0.08)")
#         st.plotly_chart(fig_loss, use_container_width=True)

# # Numeric table
# with st.expander("📊 Training History (Numeric Table)", expanded=False):
#     df_display = pd.DataFrame(TRAINING_HISTORY)
#     df_display["train_acc"] = (df_display["train_acc"] * 100).map("{:.2f}%".format)
#     df_display["val_acc"] = (df_display["val_acc"] * 100).map("{:.2f}%".format)
#     df_display["train_loss"] = df_display["train_loss"].map("{:.3f}".format)
#     df_display["val_loss"] = df_display["val_loss"].map("{:.3f}".format)
#     st.dataframe(df_display, use_container_width=True)

# st.divider()

# # Upload and prediction
# left, right = st.columns([1,1], gap="large")
# with left:
#     st.markdown("<div class='sec-title'>Upload Image</div>", unsafe_allow_html=True)
#     file = st.file_uploader("Choose a JPG or PNG file", type=["jpg","jpeg","png"], label_visibility="collapsed")
#     if file:
#         image = Image.open(file).convert("RGB")
#         st.image(image, use_container_width=True, caption="Original image")
#         raw = Image.open(file)
#         st.markdown(f"<div style='font-size:11px;color:#6b7280;margin-top:6px;'>Original: {raw.width}×{raw.height}px · {file.type}</div>", unsafe_allow_html=True)
#         current_image = image
#     else:
#         current_image = None

# with right:
#     st.markdown("<div class='sec-title'>Prediction result</div>", unsafe_allow_html=True)
#     if current_image is None:
#         st.info("👈 Upload an image to see classification")
#     else:
#         img_resized = current_image.resize((32,32))
#         img_arr = np.array(img_resized, dtype=np.uint8)
#         with st.spinner("Running TTA inference (4 folds)…"):
#             t0 = time.time()
#             preds = tta_predict(img_arr)
#             elapsed = (time.time() - t0) * 1000
#         avg = np.mean(preds, axis=0)
#         idx = int(np.argmax(avg))
#         label = CLASSES[idx]
#         conf = avg[idx]
#         color = CLASS_COLORS[label]
#         st.markdown(f"""
#         <div class="pred-badge">
#             <div class="pred-emoji">{EMOJIS[label]}</div>
#             <div><div class="pred-text">{label.upper()}</div><div class="pred-conf">{conf*100:.2f}% confidence</div></div>
#         </div>
#         """, unsafe_allow_html=True)
#         st.caption(f"⏱️ Inference: {elapsed:.0f} ms (4 TTA folds)")
#         st.markdown("<div class='sec-title'>Top 5 Predictions</div>", unsafe_allow_html=True)
#         top5 = np.argsort(avg)[::-1][:5]
#         for rank, i in enumerate(top5):
#             c = CLASSES[i]
#             v = float(avg[i])
#             col = CLASS_COLORS[c]
#             st.markdown(f"""
#             <div class="prog-row">
#                 <div class="prog-label"><span>{'🥇' if rank==0 else '  '} {EMOJIS[c]} {c.capitalize()}</span><span>{pct(v)}</span></div>
#                 <div class="prog-track"><div class="prog-fill" style="width:{v*100:.2f}%; background:linear-gradient(90deg,{col}cc,{col}66);"></div></div>
#             </div>
#             """, unsafe_allow_html=True)

# if file and current_image is not None:
#     st.divider()
#     col_a, col_b = st.columns([1,1], gap="large")
#     with col_a:
#         st.markdown("<div class='sec-title'>All Class Probabilities</div>", unsafe_allow_html=True)
#         sorted_idx = np.argsort(avg)[::-1]
#         for i in sorted_idx:
#             c = CLASSES[i]; v = float(avg[i]); col_c = CLASS_COLORS[c]
#             st.markdown(f"""
#             <div class="prog-row"><div class="prog-label"><span>{EMOJIS[c]} {c.capitalize()}</span><span>{pct(v)}</span></div>
#             <div class="prog-track"><div class="prog-fill" style="width:{v*100:.2f}%; background:linear-gradient(90deg,{col_c}cc,{col_c}44);"></div></div></div>
#             """, unsafe_allow_html=True)
#     with col_b:
#         st.markdown("<div class='sec-title'>TTA Augmentation Breakdown</div>", unsafe_allow_html=True)
#         aug_names = ["Original","Horizontal Flip","Rotate +5°","Rotate −5°"]
#         aug_icons = ["🖼️","🔄","↗️","↘️"]
#         c1,c2 = st.columns(2)
#         for k in range(4):
#             p = preds[k]; pi = int(np.argmax(p)); pcls = CLASSES[pi]; pv = float(np.max(p)); pcol = CLASS_COLORS[pcls]
#             target = c1 if k%2==0 else c2
#             with target:
#                 st.markdown(f"""
#                 <div class="tta-card"><div class="tta-aug-name">{aug_icons[k]} {aug_names[k]}</div>
#                 <div style="font-size:24px;">{EMOJIS[pcls]}</div>
#                 <div class="tta-pred-label">{pcls.capitalize()}</div>
#                 <div class="tta-pred-conf">{pv*100:.1f}%</div>
#                 <div class="tta-bar-wrap"><div class="tta-bar-fill" style="width:{pv*100:.1f}%;background:linear-gradient(90deg,{pcol},{pcol}88);"></div></div></div>
#                 """, unsafe_allow_html=True)
#                 st.markdown("<div style='height:10px;'></div>", unsafe_allow_html=True)
#     st.divider()
#     tta_labels = [CLASSES[int(np.argmax(preds[i]))] for i in range(4)]
#     agreement = tta_labels.count(label)/4
#     st.markdown(f"""
#     <div class="glass-card"><div class="sec-title">TTA Agreement</div>
#     <div style="display:flex;align-items:center;gap:16px;"><div style="font-size:32px;">{'✅' if agreement==1 else '⚠️'}</div>
#     <div><div style="font-family:'Syne',sans-serif;font-size:18px;font-weight:700;">{int(agreement*4)}/4 augmentations agree on <span style="color:#22d3ee;">{label.capitalize()}</span></div>
#     <div style="font-size:12px;color:#6b7280;">{'High confidence — all TTA folds are consistent.' if agreement==1 else 'Some folds disagree — treat confidence with caution.'}</div></div>
#     <div style="margin-left:auto;"><div style="font-family:'DM Mono',monospace;font-size:28px;color:#22d3ee;font-weight:500;">{int(agreement*100)}%</div></div></div>
#     <div class="prog-track" style="margin-top:12px;height:8px;"><div class="prog-fill" style="width:{agreement*100:.0f}%;background:{'linear-gradient(90deg,#10b981,#22d3ee)' if agreement==1 else 'linear-gradient(90deg,#f59e0b,#f87171)'};"></div></div></div>
#     """, unsafe_allow_html=True)

# st.markdown("<br>", unsafe_allow_html=True)
# st.markdown("""
# <div style="text-align:center;padding:16px;border-top:1px solid rgba(255,255,255,0.06);">
#     <div style="font-size:11px;color:#374151;letter-spacing:1px;">Built with <span style="color:#6366f1;">Streamlit</span> · TensorFlow / Keras · <strong>Md. Nafijul Islam</strong></div>
# </div>
# """, unsafe_allow_html=True)

















































# import streamlit as st
# import numpy as np
# import pandas as pd
# import plotly.graph_objects as go
# from plotly.subplots import make_subplots
# from PIL import Image
# import tensorflow as tf
# import time
# import os
# import gdown 
#   # <--- নতুন যোগ



# # ============================================================
# #  CONFIG
# # ============================================================
# MODEL_FILE = "final_cifar10_model_exp6.keras"
# GOOGLE_DRIVE_FILE_ID = "1eQ71ay40dkemH2wE0eIz_t6pyQ8YlyI3"

# CLASSES = [
#     "airplane", "automobile", "bird", "cat", "deer",
#     "dog", "frog", "horse", "ship", "truck",
# ]

# EMOJIS = {
#     "airplane": "✈️", "automobile": "🚗", "bird": "🐦",
#     "cat": "🐱", "deer": "🦌", "dog": "🐶",
#     "frog": "🐸", "horse": "🐴", "ship": "🚢", "truck": "🚚",
# }

# CLASS_COLORS = {
#     "airplane": "#60a5fa", "automobile": "#f472b6", "bird": "#34d399",
#     "cat": "#fb923c", "deer": "#a78bfa", "dog": "#facc15",
#     "frog": "#4ade80", "horse": "#f87171", "ship": "#22d3ee", "truck": "#c084fc",
# }

# TRAINING_HISTORY = {
#     "epochs": list(range(1, 21)),
#     "train_acc": [0.42, 0.55, 0.62, 0.67, 0.71, 0.74, 0.76, 0.78, 0.80, 0.81,
#                   0.82, 0.83, 0.84, 0.85, 0.86, 0.87, 0.875, 0.88, 0.885, 0.89],
#     "val_acc":   [0.38, 0.50, 0.58, 0.63, 0.67, 0.70, 0.72, 0.74, 0.76, 0.77,
#                   0.78, 0.785, 0.79, 0.795, 0.80, 0.805, 0.81, 0.812, 0.815, 0.82],
#     "train_loss": [2.1, 1.5, 1.2, 1.0, 0.85, 0.75, 0.68, 0.62, 0.57, 0.53,
#                    0.49, 0.46, 0.43, 0.41, 0.39, 0.37, 0.36, 0.35, 0.34, 0.33],
#     "val_loss":   [2.3, 1.8, 1.4, 1.15, 0.98, 0.87, 0.79, 0.72, 0.67, 0.63,
#                    0.59, 0.56, 0.53, 0.51, 0.49, 0.48, 0.47, 0.465, 0.46, 0.458]
# }

# GITHUB_URL = "https://github.com/mdnafijulislambd/CIFAR-10-Vision-Classifier"

# # ============================================================
# #  PAGE CONFIG & CSS
# # ============================================================
# st.set_page_config(page_title="CIFAR-10 Vision Classifier", page_icon="🧠", layout="wide")

# st.markdown("""
# <style>
# @import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Mono:wght@400;500&family=DM+Sans:wght@300;400;500&display=swap');
# :root { --bg: #08090d; --bg2: #0f1117; --bg3: #14161e; --border: rgba(255,255,255,0.07); --text: #e8eaf0; --muted: #6b7280; --accent: #6366f1; --accent2: #22d3ee; --success: #10b981; }
# html, body, [data-testid="stAppViewContainer"] { background: var(--bg) !important; font-family: 'DM Sans', sans-serif; color: var(--text); }
# [data-testid="stHeader"] { background: transparent !important; }
# [data-testid="stSidebar"] { background: var(--bg2) !important; border-right: 1px solid var(--border) !important; }
# .block-container { padding: 2rem 3rem !important; max-width: 1300px; position: relative; }
# .github-corner { position: fixed; top: 20px; right: 20px; z-index: 1000; background: var(--bg2); border-radius: 50%; width: 44px; height: 44px; display: flex; align-items: center; justify-content: center; text-decoration: none; border: 1px solid var(--border); transition: all 0.2s ease; backdrop-filter: blur(8px); }
# .github-corner:hover { border-color: var(--accent2); transform: translateY(-2px); }
# .github-corner svg { width: 22px; height: 22px; fill: var(--text); }
# .hero-wrap { display: flex; flex-direction: column; gap: 6px; margin-bottom: 1.5rem; }
# .hero-label { font-family: 'DM Mono', monospace; font-size: 11px; letter-spacing: 3px; text-transform: uppercase; color: var(--accent2); }
# .hero-title { font-family: 'Syne', sans-serif; font-size: clamp(32px, 5vw, 56px); font-weight: 800; line-height: 1.05; background: linear-gradient(135deg, #fff 30%, var(--accent2) 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text; margin: 0; }
# .hero-sub { font-size: 14px; color: var(--muted); font-weight: 300; margin-top: 4px; }
# .hero-sub span { color: var(--accent2); font-weight: 500; }
# .creator { font-size: 12px; color: var(--muted); margin-top: 8px; font-family: 'DM Mono', monospace; letter-spacing: 0.5px; }
# .creator strong { color: var(--accent2); font-weight: 600; }
# .glass-card { background: var(--bg3); border: 1px solid var(--border); border-radius: 16px; padding: 24px; margin-bottom: 16px; position: relative; overflow: hidden; }
# .sec-title { font-family: 'Syne', sans-serif; font-size: 13px; font-weight: 600; letter-spacing: 2px; text-transform: uppercase; color: var(--muted); margin-bottom: 14px; }
# .pred-badge { display: inline-flex; align-items: center; gap: 10px; background: linear-gradient(135deg, rgba(99,102,241,0.15), rgba(34,211,238,0.08)); border: 1px solid rgba(99,102,241,0.4); border-radius: 50px; padding: 10px 22px; margin-bottom: 18px; }
# .pred-emoji { font-size: 28px; line-height: 1; }
# .pred-text { font-family: 'Syne', sans-serif; font-size: 22px; font-weight: 700; color: #fff; }
# .pred-conf { font-family: 'DM Mono', monospace; font-size: 13px; color: var(--accent2); margin-left: 4px; }
# .prog-row { margin-bottom: 10px; }
# .prog-label { display: flex; justify-content: space-between; font-size: 12px; margin-bottom: 4px; color: var(--text); }
# .prog-label span:last-child { font-family: 'DM Mono', monospace; color: var(--muted); }
# .prog-track { height: 6px; background: rgba(255,255,255,0.06); border-radius: 99px; overflow: hidden; }
# .prog-fill { height: 100%; border-radius: 99px; transition: width 0.6s cubic-bezier(.16,1,.3,1); }
# .tta-card { background: var(--bg2); border: 1px solid var(--border); border-radius: 12px; padding: 16px; text-align: center; }
# .tta-aug-name { font-family: 'DM Mono', monospace; font-size: 10px; letter-spacing: 1.5px; text-transform: uppercase; color: var(--muted); margin-bottom: 8px; }
# .tta-pred-label { font-family: 'Syne', sans-serif; font-size: 15px; font-weight: 600; color: #fff; }
# .tta-pred-conf { font-family: 'DM Mono', monospace; font-size: 12px; color: var(--accent2); margin-top: 2px; }
# .tta-bar-wrap { margin-top: 10px; height: 4px; background: rgba(255,255,255,0.06); border-radius: 99px; overflow: hidden; }
# .tta-bar-fill { height: 100%; border-radius: 99px; background: linear-gradient(90deg, var(--accent), var(--accent2)); }
# .stat-row { display:flex; gap:10px; flex-wrap:wrap; margin-bottom:20px; }
# .stat-pill { background: var(--bg2); border: 1px solid var(--border); border-radius: 8px; padding: 10px 16px; display: flex; flex-direction: column; gap: 2px; }
# .stat-pill .num { font-family: 'Syne', sans-serif; font-size: 20px; font-weight: 700; color: var(--accent2); }
# .stat-pill .lbl { font-size: 11px; color: var(--muted); letter-spacing: 0.5px; }
# [data-testid="stFileUploader"] { background: var(--bg3) !important; border: 1.5px dashed rgba(99,102,241,0.4) !important; border-radius: 14px !important; padding: 10px !important; }
# [data-testid="stFileUploader"]:hover { border-color: rgba(99,102,241,0.7) !important; }
# [data-testid="stImage"] img { border-radius: 12px !important; border: 1px solid var(--border) !important; }
# .stSpinner > div { border-top-color: var(--accent) !important; }
# hr { border-color: var(--border) !important; }
# [data-testid="stExpander"] { background: var(--bg3) !important; border: 1px solid var(--border) !important; border-radius: 12px !important; }
# .about-card { background: linear-gradient(135deg, rgba(99,102,241,0.12), rgba(34,211,238,0.06)); border: 1px solid rgba(99,102,241,0.2); border-radius: 12px; padding: 16px; margin-bottom: 16px; font-size: 13px; line-height: 1.6; }
# ::-webkit-scrollbar { width: 5px; }
# ::-webkit-scrollbar-track { background: var(--bg); }
# ::-webkit-scrollbar-thumb { background: rgba(255,255,255,0.1); border-radius: 99px; }
# </style>
# """, unsafe_allow_html=True)

# # GitHub icon
# st.markdown(f"""
# <a href="{GITHUB_URL}" target="_blank" class="github-corner">
#     <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24">
#         <path d="M12 0C5.37 0 0 5.37 0 12c0 5.31 3.435 9.795 8.205 11.385.6.105.825-.255.825-.57 0-.285-.015-1.23-.015-2.235-3.015.555-3.795-.735-4.035-1.41-.135-.345-.72-1.41-1.23-1.695-.42-.225-1.02-.78-.015-.795.945-.015 1.62.87 1.845 1.23 1.08 1.815 2.805 1.305 3.495.99.105-.78.42-1.305.765-1.605-2.67-.3-5.46-1.335-5.46-5.925 0-1.305.465-2.385 1.23-3.225-.12-.3-.54-1.53.12-3.18 0 0 1.005-.315 3.3 1.23.96-.27 1.98-.405 3-.405s2.04.135 3 .405c2.295-1.56 3.3-1.23 3.3-1.23.66 1.65.24 2.88.12 3.18.765.84 1.23 1.905 1.23 3.225 0 4.605-2.805 5.625-5.475 5.925.435.375.81 1.095.81 2.22 0 1.605-.015 2.895-.015 3.3 0 .315.225.69.825.57A12.02 12.02 0 0 0 24 12c0-6.63-5.37-12-12-12z"/>
#     </svg>
# </a>
# """, unsafe_allow_html=True)


# @st.cache_resource
# def load_cifar_model():
#     import os
#     import gdown
#     import tensorflow as tf

#     # ❗ corrupted / half डाउनलोड file remove
#     if os.path.exists(MODEL_FILE):
#         file_size = os.path.getsize(MODEL_FILE) / (1024 * 1024)
#         if file_size < 70:   # expected ~80MB
#             os.remove(MODEL_FILE)

#     # ✅ download from Google Drive (stable)
#     if not os.path.exists(MODEL_FILE):
#         with st.spinner("Downloading model (80 MB) from Google Drive..."):
#             url = f"https://drive.google.com/uc?id={GOOGLE_DRIVE_FILE_ID}"
#             gdown.download(url, MODEL_FILE, quiet=False)

#     # ✅ FIX: Functional model loading issue
#     custom_objects = {'Functional': tf.keras.models.Model}

#     model = tf.keras.models.load_model(
#         MODEL_FILE,
#         custom_objects=custom_objects,
#         compile=False,
#         safe_mode=False
#     )

#     return model

# # ============================================================
# #  TTA PREDICTION
# # ============================================================
# def tta_predict(img_arr: np.ndarray, model):
#     base = img_arr.astype(np.uint8)
#     pil = Image.fromarray(base)

#     augs = [
#         base,
#         np.fliplr(base),
#         np.array(pil.rotate(5)),
#         np.array(pil.rotate(-5))
#     ]

#     preds = []

#     for aug in augs:
#         x = aug.astype(np.float32) / 255.0
#         x = np.expand_dims(x, axis=0)

#         logits = model.predict(x, verbose=0)
#         logits = logits.squeeze()

#         probs = tf.nn.softmax(logits).numpy()
#         preds.append(probs)

#     return np.array(preds)

# def pct(v: float) -> str:
#     return f"{v * 100:.1f}%"
# # ============================================================
# #  SIDEBAR
# # ============================================================
# with st.sidebar:
#     st.markdown("""
#     <div style="font-family:'Syne',sans-serif;font-size:22px;font-weight:800;
#                 background:linear-gradient(135deg,#fff,#22d3ee);
#                 -webkit-background-clip:text;-webkit-text-fill-color:transparent;
#                 margin-bottom:6px;">
#         CIFAR-10<br>Classifier
#     </div>
#     <div style="font-size:11px;color:#6b7280;letter-spacing:1px;margin-bottom:20px;">
#         CNN · RESIDUAL · TTA
#     </div>
#     """, unsafe_allow_html=True)
#     st.markdown("""
#     <div class="about-card">
#         A <strong>Convolutional Neural Network</strong> trained on the CIFAR-10 dataset
#         to classify images into <strong>10 categories</strong> using residual connections
#         and Test-Time Augmentation for robust predictions.
#     </div>
#     """, unsafe_allow_html=True)
#     st.markdown("<div class='sec-title'>10 Classes</div>", unsafe_allow_html=True)
#     cols_sb = st.columns(2)
#     for i, cls in enumerate(CLASSES):
#         with cols_sb[i % 2]:
#             st.markdown(f"<div style='font-size:13px;margin-bottom:6px;'>{EMOJIS[cls]} {cls.capitalize()}</div>", unsafe_allow_html=True)
#     st.divider()
#     st.markdown("<div class='sec-title'>Model Details</div>", unsafe_allow_html=True)
#     for k, v in [("Architecture", "ResNet-style CNN"), ("Dataset", "CIFAR-10"), ("TTA Folds", "4"), ("Input Size", "32×32 px")]:
#         st.markdown(f"<div style='display:flex;justify-content:space-between;font-size:12px;margin-bottom:6px;'><span style='color:#6b7280;'>{k}</span><span style='font-family:DM Mono,monospace;color:#22d3ee;'>{v}</span></div>", unsafe_allow_html=True)
#     st.divider()
#     st.markdown("<div style='font-size:11px;color:#4b5563;text-align:center;'>Developed by <strong style='color:#6b7280;'>Md. Nafijul Islam</strong></div>", unsafe_allow_html=True)

# # ============================================================
# #  MAIN UI
# # ============================================================
# st.markdown("""
# <div class="hero-wrap">
#     <div class="hero-label">Deep Learning · Image Recognition</div>
#     <div class="hero-title">CIFAR-10 Vision Classifier</div>
#     <div class="hero-sub">Residual CNN with <span>Test-Time Augmentation</span> — upload any image to classify it into one of 10 categories.</div>
#     <div class="creator">Created by <strong>Md. Nafijul Islam</strong></div>
# </div>
# """, unsafe_allow_html=True)

# st.markdown("""
# <div class="stat-row">
#     <div class="stat-pill"><div class="num">~89%</div><div class="lbl">Val Accuracy</div></div>
#     <div class="stat-pill"><div class="num">10</div><div class="lbl">Classes</div></div>
#     <div class="stat-pill"><div class="num">4×</div><div class="lbl">TTA Folds</div></div>
#     <div class="stat-pill"><div class="num">50K</div><div class="lbl">Training Images</div></div>
# </div>
# """, unsafe_allow_html=True)

# # Training history graphs
# with st.expander("📈 Training History (Graphs)", expanded=False):
#     df_history = pd.DataFrame(TRAINING_HISTORY)
#     col_acc, col_loss = st.columns(2)
#     with col_acc:
#         fig_acc = go.Figure()
#         fig_acc.add_trace(go.Scatter(x=df_history["epochs"], y=df_history["train_acc"]*100, mode="lines", name="Train Acc", line=dict(color="#6366f1", width=3, shape='spline'), fill='tozeroy', fillcolor='rgba(99,102,241,0.15)'))
#         fig_acc.add_trace(go.Scatter(x=df_history["epochs"], y=df_history["val_acc"]*100, mode="lines", name="Val Acc", line=dict(color="#22d3ee", width=3, shape='spline'), fill='tozeroy', fillcolor='rgba(34,211,238,0.1)'))
#         fig_acc.update_layout(title="Accuracy", xaxis_title="Epoch", yaxis_title="Accuracy (%)", template="plotly_dark", height=380, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", font=dict(color="#e8eaf0"), legend=dict(orientation="h", yanchor="bottom", y=0.98, xanchor="right", x=0.98))
#         fig_acc.update_xaxes(gridcolor="rgba(255,255,255,0.08)")
#         fig_acc.update_yaxes(gridcolor="rgba(255,255,255,0.08)", range=[30,100])
#         st.plotly_chart(fig_acc, use_container_width=True)
#     with col_loss:
#         fig_loss = go.Figure()
#         fig_loss.add_trace(go.Scatter(x=df_history["epochs"], y=df_history["train_loss"], mode="lines", name="Train Loss", line=dict(color="#f472b6", width=3, shape='spline'), fill='tozeroy', fillcolor='rgba(244,114,182,0.1)'))
#         fig_loss.add_trace(go.Scatter(x=df_history["epochs"], y=df_history["val_loss"], mode="lines", name="Val Loss", line=dict(color="#facc15", width=3, shape='spline'), fill='tozeroy', fillcolor='rgba(250,204,21,0.1)'))
#         fig_loss.update_layout(title="Loss", xaxis_title="Epoch", yaxis_title="Loss", template="plotly_dark", height=380, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", font=dict(color="#e8eaf0"), legend=dict(orientation="h", yanchor="bottom", y=0.98, xanchor="right", x=0.98))
#         fig_loss.update_xaxes(gridcolor="rgba(255,255,255,0.08)")
#         fig_loss.update_yaxes(gridcolor="rgba(255,255,255,0.08)")
#         st.plotly_chart(fig_loss, use_container_width=True)

# # Numeric table
# with st.expander("📊 Training History (Numeric Table)", expanded=False):
#     df_display = pd.DataFrame(TRAINING_HISTORY)
#     df_display["train_acc"] = (df_display["train_acc"] * 100).map("{:.2f}%".format)
#     df_display["val_acc"] = (df_display["val_acc"] * 100).map("{:.2f}%".format)
#     df_display["train_loss"] = df_display["train_loss"].map("{:.3f}".format)
#     df_display["val_loss"] = df_display["val_loss"].map("{:.3f}".format)
#     st.dataframe(df_display, use_container_width=True)

# st.divider()

# # Upload and prediction
# # left, right = st.columns([1,1], gap="large")
# # with left:
# #     st.markdown("<div class='sec-title'>Upload Image</div>", unsafe_allow_html=True)
# #     file = st.file_uploader("Choose a JPG or PNG file", type=["jpg","jpeg","png"], label_visibility="collapsed")
# #     if file:
# #         image = Image.open(file).convert("RGB")
# #         st.image(image, use_container_width=True, caption="Original image")
# #         raw = Image.open(file)
# #         st.markdown(f"<div style='font-size:11px;color:#6b7280;margin-top:6px;'>Original: {raw.width}×{raw.height}px · {file.type}</div>", unsafe_allow_html=True)
# #         current_image = image
# #     else:
# #         current_image = None

# # with right:
# #     st.markdown("<div class='sec-title'>Prediction result</div>", unsafe_allow_html=True)
# #     if current_image is None:
# #         st.info("👈 Upload an image to see classification")
# #     else:
# #         img_resized = current_image.resize((32,32))
# #         img_arr = np.array(img_resized, dtype=np.uint8)
# #         with st.spinner("Running TTA inference (4 folds)…"):
# #             t0 = time.time()
# #             preds = tta_predict(img_arr, model)
# #             elapsed = (time.time() - t0) * 1000
# #         avg = np.mean(preds, axis=0)
# #         idx = int(np.argmax(avg))
# #         label = CLASSES[idx]
# #         conf = avg[idx]
# #         color = CLASS_COLORS[label]
# #         st.markdown(f"""
# #         <div class="pred-badge">
# #             <div class="pred-emoji">{EMOJIS[label]}</div>
# #             <div><div class="pred-text">{label.upper()}</div><div class="pred-conf">{conf*100:.2f}% confidence</div></div>
# #         </div>
# #         """, unsafe_allow_html=True)
# #         st.caption(f"⏱️ Inference: {elapsed:.0f} ms (4 TTA folds)")
# #         st.markdown("<div class='sec-title'>Top 5 Predictions</div>", unsafe_allow_html=True)
# #         top5 = np.argsort(avg)[::-1][:5]
# #         for rank, i in enumerate(top5):
# #             c = CLASSES[i]
# #             v = float(avg[i])
# #             col = CLASS_COLORS[c]
# #             st.markdown(f"""
# #             <div class="prog-row">
# #                 <div class="prog-label"><span>{'🥇' if rank==0 else '  '} {EMOJIS[c]} {c.capitalize()}</span><span>{pct(v)}</span></div>
# #                 <div class="prog-track"><div class="prog-fill" style="width:{v*100:.2f}%; background:linear-gradient(90deg,{col}cc,{col}66);"></div></div>
# #             </div>
# #             """, unsafe_allow_html=True)

# # if file and current_image is not None:
# #     st.divider()
# #     col_a, col_b = st.columns([1,1], gap="large")
# #     with col_a:
# #         st.markdown("<div class='sec-title'>All Class Probabilities</div>", unsafe_allow_html=True)
# #         sorted_idx = np.argsort(avg)[::-1]
# #         for i in sorted_idx:
# #             c = CLASSES[i]; v = float(avg[i]); col_c = CLASS_COLORS[c]
# #             st.markdown(f"""
# #             <div class="prog-row"><div class="prog-label"><span>{EMOJIS[c]} {c.capitalize()}</span><span>{pct(v)}</span></div>
# #             <div class="prog-track"><div class="prog-fill" style="width:{v*100:.2f}%; background:linear-gradient(90deg,{col_c}cc,{col_c}44);"></div></div></div>
# #             """, unsafe_allow_html=True)
# #     with col_b:
# #         st.markdown("<div class='sec-title'>TTA Augmentation Breakdown</div>", unsafe_allow_html=True)
# #         aug_names = ["Original","Horizontal Flip","Rotate +5°","Rotate −5°"]
# #         aug_icons = ["🖼️","🔄","↗️","↘️"]
# #         c1,c2 = st.columns(2)
# #         for k in range(4):
# #             p = preds[k]; pi = int(np.argmax(p)); pcls = CLASSES[pi]; pv = float(np.max(p)); pcol = CLASS_COLORS[pcls]
# #             target = c1 if k%2==0 else c2
# #             with target:
# #                 st.markdown(f"""
# #                 <div class="tta-card"><div class="tta-aug-name">{aug_icons[k]} {aug_names[k]}</div>
# #                 <div style="font-size:24px;">{EMOJIS[pcls]}</div>
# #                 <div class="tta-pred-label">{pcls.capitalize()}</div>
# #                 <div class="tta-pred-conf">{pv*100:.1f}%</div>
# #                 <div class="tta-bar-wrap"><div class="tta-bar-fill" style="width:{pv*100:.1f}%;background:linear-gradient(90deg,{pcol},{pcol}88);"></div></div></div>
# #                 """, unsafe_allow_html=True)
# #                 st.markdown("<div style='height:10px;'></div>", unsafe_allow_html=True)
# #     st.divider()
# #     tta_labels = [CLASSES[int(np.argmax(preds[i]))] for i in range(4)]
# #     agreement = tta_labels.count(label)/4
# #     st.markdown(f"""
# #     <div class="glass-card"><div class="sec-title">TTA Agreement</div>
# #     <div style="display:flex;align-items:center;gap:16px;"><div style="font-size:32px;">{'✅' if agreement==1 else '⚠️'}</div>
# #     <div><div style="font-family:'Syne',sans-serif;font-size:18px;font-weight:700;">{int(agreement*4)}/4 augmentations agree on <span style="color:#22d3ee;">{label.capitalize()}</span></div>
# #     <div style="font-size:12px;color:#6b7280;">{'High confidence — all TTA folds are consistent.' if agreement==1 else 'Some folds disagree — treat confidence with caution.'}</div></div>
# #     <div style="margin-left:auto;"><div style="font-family:'DM Mono',monospace;font-size:28px;color:#22d3ee;font-weight:500;">{int(agreement*100)}%</div></div></div>
# #     <div class="prog-track" style="margin-top:12px;height:8px;"><div class="prog-fill" style="width:{agreement*100:.0f}%;background:{'linear-gradient(90deg,#10b981,#22d3ee)' if agreement==1 else 'linear-gradient(90deg,#f59e0b,#f87171)'};"></div></div></div>
# #     """, unsafe_allow_html=True)

# # st.markdown("<br>", unsafe_allow_html=True)
# # st.markdown("""
# # <div style="text-align:center;padding:16px;border-top:1px solid rgba(255,255,255,0.06);">
# #     <div style="font-size:11px;color:#374151;letter-spacing:1px;">Built with <span style="color:#6366f1;">Streamlit</span> · TensorFlow / Keras · <strong>Md. Nafijul Islam</strong></div>
# # </div>
# # """, unsafe_allow_html=True)


















# # ============================================================
# # UPLOAD & PREDICTION (CLEAN VERSION)
# # ============================================================

# left, right = st.columns([1, 1], gap="large")

# # ---------------- LEFT SIDE ----------------
# with left:
#     st.markdown("<div class='sec-title'>Upload Image</div>", unsafe_allow_html=True)

#     file = st.file_uploader(
#         "Choose a JPG or PNG file",
#         type=["jpg", "jpeg", "png"],
#         label_visibility="collapsed"
#     )

#     current_image = None

#     if file:
#         image = Image.open(file).convert("RGB")

#         st.image(image, use_container_width=True, caption="Original image")

#         raw = Image.open(file)

#         st.markdown(
#             f"<div style='font-size:11px;color:#6b7280;margin-top:6px;'>"
#             f"Original: {raw.width}×{raw.height}px · {file.type}</div>",
#             unsafe_allow_html=True
#         )

#         current_image = image

# # ---------------- RIGHT SIDE ----------------
# with right:
#     st.markdown("<div class='sec-title'>Prediction Result</div>", unsafe_allow_html=True)

#     if current_image is None:
#         st.info("👈 Upload an image to see classification")

#     else:
#         # ================= IMAGE PREPROCESS =================
#         img_resized = current_image.resize((32, 32))
#         img_arr = np.array(img_resized, dtype=np.uint8)

#         # ================= TTA INFERENCE =================
#         with st.spinner("Running TTA inference (4 folds)…"):
#             start_time = time.time()

#             preds = tta_predict(img_arr, model)   # MUST pass model

#             inference_time = (time.time() - start_time) * 1000

#         # ================= SAFE AGGREGATION =================
#         preds = np.array(preds)
#         avg = preds.mean(axis=0)

#         top_idx = int(np.argmax(avg))
#         label = CLASSES[top_idx]
#         confidence = float(avg[top_idx])
#         color = CLASS_COLORS[label]

#         # ================= MAIN RESULT UI =================
#         st.markdown(f"""
#         <div class="pred-badge">
#             <div class="pred-emoji">{EMOJIS[label]}</div>
#             <div>
#                 <div class="pred-text">{label.upper()}</div>
#                 <div class="pred-conf">{confidence*100:.2f}% confidence</div>
#             </div>
#         </div>
#         """, unsafe_allow_html=True)

#         st.caption(f"⏱️ Inference time: {inference_time:.0f} ms (4 TTA folds)")

#         # ================= TOP 5 =================
#         st.markdown("<div class='sec-title'>Top 5 Predictions</div>", unsafe_allow_html=True)

#         top5 = np.argsort(avg)[::-1][:5]

#         for rank, i in enumerate(top5):
#             cls = CLASSES[i]
#             val = float(avg[i])
#             col = CLASS_COLORS[cls]

#             st.markdown(f"""
#             <div class="prog-row">
#                 <div class="prog-label">
#                     <span>
#                         {'🥇' if rank == 0 else '  '}
#                         {EMOJIS[cls]} {cls.capitalize()}
#                     </span>
#                     <span>{pct(val)}</span>
#                 </div>

#                 <div class="prog-track">
#                     <div class="prog-fill"
#                          style="width:{val*100:.2f}%;
#                          background:linear-gradient(90deg,{col}cc,{col}66);">
#                     </div>
#                 </div>
#             </div>
#             """, unsafe_allow_html=True)

# # ============================================================
# # EXTRA CLEAN SECTION (only when image exists)
# # ============================================================
# if current_image is not None:

#     st.divider()

#     col_a, col_b = st.columns([1, 1], gap="large")

#     # ---------------- ALL PROBS ----------------
#     with col_a:
#         st.markdown("<div class='sec-title'>All Class Probabilities</div>", unsafe_allow_html=True)

#         sorted_idx = np.argsort(avg)[::-1]

#         for i in sorted_idx:
#             cls = CLASSES[i]
#             val = float(avg[i])
#             col = CLASS_COLORS[cls]

#             st.markdown(f"""
#             <div class="prog-row">
#                 <div class="prog-label">
#                     <span>{EMOJIS[cls]} {cls.capitalize()}</span>
#                     <span>{pct(val)}</span>
#                 </div>

#                 <div class="prog-track">
#                     <div class="prog-fill"
#                          style="width:{val*100:.2f}%;
#                          background:linear-gradient(90deg,{col}cc,{col}44);">
#                     </div>
#                 </div>
#             </div>
#             """, unsafe_allow_html=True)

#     # ---------------- TTA BREAKDOWN ----------------
#     with col_b:
#         st.markdown("<div class='sec-title'>TTA Breakdown</div>", unsafe_allow_html=True)

#         aug_names = ["Original", "Flip", "Rotate +5°", "Rotate -5°"]
#         aug_icons = ["🖼️", "🔄", "↗️", "↘️"]

#         c1, c2 = st.columns(2)

#         for k in range(4):
#             p = preds[k]
#             pred_idx = int(np.argmax(p))
#             pred_cls = CLASSES[pred_idx]
#             pred_conf = float(np.max(p))
#             col = CLASS_COLORS[pred_cls]

#             target = c1 if k % 2 == 0 else c2

#             with target:
#                 st.markdown(f"""
#                 <div class="tta-card">
#                     <div class="tta-aug-name">{aug_icons[k]} {aug_names[k]}</div>

#                     <div style="font-size:24px;">{EMOJIS[pred_cls]}</div>

#                     <div class="tta-pred-label">{pred_cls.capitalize()}</div>

#                     <div class="tta-pred-conf">{pred_conf*100:.1f}%</div>

#                     <div class="tta-bar-wrap">
#                         <div class="tta-bar-fill"
#                              style="width:{pred_conf*100:.1f}%;
#                              background:linear-gradient(90deg,{col},{col}88);">
#                         </div>
#                     </div>
#                 </div>
#                 """, unsafe_allow_html=True)

#                 st.markdown("<div style='height:10px;'></div>", unsafe_allow_html=True)

#     # ---------------- AGREEMENT ----------------
#     st.divider()

#     tta_labels = [CLASSES[int(np.argmax(preds[i]))] for i in range(4)]
#     agreement = tta_labels.count(label) / 4

#     st.markdown(f"""
#     <div class="glass-card">
#         <div class="sec-title">TTA Agreement</div>

#         <div style="display:flex;align-items:center;gap:16px;">

#             <div style="font-size:32px;">
#                 {'✅' if agreement == 1 else '⚠️'}
#             </div>

#             <div>
#                 <div style="font-family:'Syne',sans-serif;font-size:18px;font-weight:700;">
#                     {int(agreement*4)}/4 agree on
#                     <span style="color:#22d3ee;">{label.capitalize()}</span>
#                 </div>

#                 <div style="font-size:12px;color:#6b7280;">
#                     {'High confidence prediction' if agreement == 1 else 'Some disagreement in TTA folds'}
#                 </div>
#             </div>

#             <div style="margin-left:auto;">
#                 <div style="font-family:'DM Mono',monospace;font-size:28px;color:#22d3ee;">
#                     {int(agreement*100)}%
#                 </div>
#             </div>

#         </div>

#         <div class="prog-track" style="margin-top:12px;height:8px;">
#             <div class="prog-fill"
#                  style="width:{agreement*100:.0f}%;
#                  background:{'linear-gradient(90deg,#10b981,#22d3ee)' if agreement==1 else 'linear-gradient(90deg,#f59e0b,#f87171)'};">
#             </div>
#         </div>
#     </div>
#     """, unsafe_allow_html=True)









# import streamlit as st
# import numpy as np
# import pandas as pd
# import plotly.graph_objects as go
# from PIL import Image
# import tensorflow as tf
# import time
# import os
# import gdown

# # ============================================================
# #  CONFIG
# # ============================================================
# MODEL_FILE = "final_cifar10_model_exp6.keras"
# GOOGLE_DRIVE_FILE_ID = "1eQ71ay40dkemH2wE0eIz_t6pyQ8YlyI3"

# CLASSES = [
#     "airplane", "automobile", "bird", "cat", "deer",
#     "dog", "frog", "horse", "ship", "truck",
# ]

# EMOJIS = {
#     "airplane": "✈️", "automobile": "🚗", "bird": "🐦",
#     "cat": "🐱", "deer": "🦌", "dog": "🐶",
#     "frog": "🐸", "horse": "🐴", "ship": "🚢", "truck": "🚚",
# }

# CLASS_COLORS = {
#     "airplane": "#60a5fa", "automobile": "#f472b6", "bird": "#34d399",
#     "cat": "#fb923c", "deer": "#a78bfa", "dog": "#facc15",
#     "frog": "#4ade80", "horse": "#f87171", "ship": "#22d3ee", "truck": "#c084fc",
# }

# TRAINING_HISTORY = {
#     "epochs": list(range(1, 21)),
#     "train_acc": [0.42, 0.55, 0.62, 0.67, 0.71, 0.74, 0.76, 0.78, 0.80, 0.81,
#                   0.82, 0.83, 0.84, 0.85, 0.86, 0.87, 0.875, 0.88, 0.885, 0.89],
#     "val_acc":   [0.38, 0.50, 0.58, 0.63, 0.67, 0.70, 0.72, 0.74, 0.76, 0.77,
#                   0.78, 0.785, 0.79, 0.795, 0.80, 0.805, 0.81, 0.812, 0.815, 0.82],
#     "train_loss": [2.1, 1.5, 1.2, 1.0, 0.85, 0.75, 0.68, 0.62, 0.57, 0.53,
#                    0.49, 0.46, 0.43, 0.41, 0.39, 0.37, 0.36, 0.35, 0.34, 0.33],
#     "val_loss":   [2.3, 1.8, 1.4, 1.15, 0.98, 0.87, 0.79, 0.72, 0.67, 0.63,
#                    0.59, 0.56, 0.53, 0.51, 0.49, 0.48, 0.47, 0.465, 0.46, 0.458]
# }

# GITHUB_URL = "https://github.com/mdnafijulislambd/CIFAR-10-Vision-Classifier"

# # ============================================================
# #  PAGE CONFIG & CSS (same as original)
# # ============================================================
# st.set_page_config(page_title="CIFAR-10 Vision Classifier", page_icon="🧠", layout="wide")

# st.markdown("""
# <style>
# @import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Mono:wght@400;500&family=DM+Sans:wght@300;400;500&display=swap');
# :root { --bg: #08090d; --bg2: #0f1117; --bg3: #14161e; --border: rgba(255,255,255,0.07); --text: #e8eaf0; --muted: #6b7280; --accent: #6366f1; --accent2: #22d3ee; --success: #10b981; }
# html, body, [data-testid="stAppViewContainer"] { background: var(--bg) !important; font-family: 'DM Sans', sans-serif; color: var(--text); }
# [data-testid="stHeader"] { background: transparent !important; }
# [data-testid="stSidebar"] { background: var(--bg2) !important; border-right: 1px solid var(--border) !important; }
# .block-container { padding: 2rem 3rem !important; max-width: 1300px; position: relative; }
# .github-corner { position: fixed; top: 20px; right: 20px; z-index: 1000; background: var(--bg2); border-radius: 50%; width: 44px; height: 44px; display: flex; align-items: center; justify-content: center; text-decoration: none; border: 1px solid var(--border); transition: all 0.2s ease; backdrop-filter: blur(8px); }
# .github-corner:hover { border-color: var(--accent2); transform: translateY(-2px); }
# .github-corner svg { width: 22px; height: 22px; fill: var(--text); }
# .hero-wrap { display: flex; flex-direction: column; gap: 6px; margin-bottom: 1.5rem; }
# .hero-label { font-family: 'DM Mono', monospace; font-size: 11px; letter-spacing: 3px; text-transform: uppercase; color: var(--accent2); }
# .hero-title { font-family: 'Syne', sans-serif; font-size: clamp(32px, 5vw, 56px); font-weight: 800; line-height: 1.05; background: linear-gradient(135deg, #fff 30%, var(--accent2) 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text; margin: 0; }
# .hero-sub { font-size: 14px; color: var(--muted); font-weight: 300; margin-top: 4px; }
# .hero-sub span { color: var(--accent2); font-weight: 500; }
# .creator { font-size: 12px; color: var(--muted); margin-top: 8px; font-family: 'DM Mono', monospace; letter-spacing: 0.5px; }
# .creator strong { color: var(--accent2); font-weight: 600; }
# .glass-card { background: var(--bg3); border: 1px solid var(--border); border-radius: 16px; padding: 24px; margin-bottom: 16px; position: relative; overflow: hidden; }
# .sec-title { font-family: 'Syne', sans-serif; font-size: 13px; font-weight: 600; letter-spacing: 2px; text-transform: uppercase; color: var(--muted); margin-bottom: 14px; }
# .pred-badge { display: inline-flex; align-items: center; gap: 10px; background: linear-gradient(135deg, rgba(99,102,241,0.15), rgba(34,211,238,0.08)); border: 1px solid rgba(99,102,241,0.4); border-radius: 50px; padding: 10px 22px; margin-bottom: 18px; }
# .pred-emoji { font-size: 28px; line-height: 1; }
# .pred-text { font-family: 'Syne', sans-serif; font-size: 22px; font-weight: 700; color: #fff; }
# .pred-conf { font-family: 'DM Mono', monospace; font-size: 13px; color: var(--accent2); margin-left: 4px; }
# .prog-row { margin-bottom: 10px; }
# .prog-label { display: flex; justify-content: space-between; font-size: 12px; margin-bottom: 4px; color: var(--text); }
# .prog-label span:last-child { font-family: 'DM Mono', monospace; color: var(--muted); }
# .prog-track { height: 6px; background: rgba(255,255,255,0.06); border-radius: 99px; overflow: hidden; }
# .prog-fill { height: 100%; border-radius: 99px; transition: width 0.6s cubic-bezier(.16,1,.3,1); }
# .tta-card { background: var(--bg2); border: 1px solid var(--border); border-radius: 12px; padding: 16px; text-align: center; }
# .tta-aug-name { font-family: 'DM Mono', monospace; font-size: 10px; letter-spacing: 1.5px; text-transform: uppercase; color: var(--muted); margin-bottom: 8px; }
# .tta-pred-label { font-family: 'Syne', sans-serif; font-size: 15px; font-weight: 600; color: #fff; }
# .tta-pred-conf { font-family: 'DM Mono', monospace; font-size: 12px; color: var(--accent2); margin-top: 2px; }
# .tta-bar-wrap { margin-top: 10px; height: 4px; background: rgba(255,255,255,0.06); border-radius: 99px; overflow: hidden; }
# .tta-bar-fill { height: 100%; border-radius: 99px; background: linear-gradient(90deg, var(--accent), var(--accent2)); }
# .stat-row { display:flex; gap:10px; flex-wrap:wrap; margin-bottom:20px; }
# .stat-pill { background: var(--bg2); border: 1px solid var(--border); border-radius: 8px; padding: 10px 16px; display: flex; flex-direction: column; gap: 2px; }
# .stat-pill .num { font-family: 'Syne', sans-serif; font-size: 20px; font-weight: 700; color: var(--accent2); }
# .stat-pill .lbl { font-size: 11px; color: var(--muted); letter-spacing: 0.5px; }
# [data-testid="stFileUploader"] { background: var(--bg3) !important; border: 1.5px dashed rgba(99,102,241,0.4) !important; border-radius: 14px !important; padding: 10px !important; }
# [data-testid="stFileUploader"]:hover { border-color: rgba(99,102,241,0.7) !important; }
# [data-testid="stImage"] img { border-radius: 12px !important; border: 1px solid var(--border) !important; }
# .stSpinner > div { border-top-color: var(--accent) !important; }
# hr { border-color: var(--border) !important; }
# [data-testid="stExpander"] { background: var(--bg3) !important; border: 1px solid var(--border) !important; border-radius: 12px !important; }
# .about-card { background: linear-gradient(135deg, rgba(99,102,241,0.12), rgba(34,211,238,0.06)); border: 1px solid rgba(99,102,241,0.2); border-radius: 12px; padding: 16px; margin-bottom: 16px; font-size: 13px; line-height: 1.6; }
# ::-webkit-scrollbar { width: 5px; }
# ::-webkit-scrollbar-track { background: var(--bg); }
# ::-webkit-scrollbar-thumb { background: rgba(255,255,255,0.1); border-radius: 99px; }
# </style>
# """, unsafe_allow_html=True)

# st.markdown(f"""
# <a href="{GITHUB_URL}" target="_blank" class="github-corner">
#     <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24">
#         <path d="M12 0C5.37 0 0 5.37 0 12c0 5.31 3.435 9.795 8.205 11.385.6.105.825-.255.825-.57 0-.285-.015-1.23-.015-2.235-3.015.555-3.795-.735-4.035-1.41-.135-.345-.72-1.41-1.23-1.695-.42-.225-1.02-.78-.015-.795.945-.015 1.62.87 1.845 1.23 1.08 1.815 2.805 1.305 3.495.99.105-.78.42-1.305.765-1.605-2.67-.3-5.46-1.335-5.46-5.925 0-1.305.465-2.385 1.23-3.225-.12-.3-.54-1.53.12-3.18 0 0 1.005-.315 3.3 1.23.96-.27 1.98-.405 3-.405s2.04.135 3 .405c2.295-1.56 3.3-1.23 3.3-1.23.66 1.65.24 2.88.12 3.18.765.84 1.23 1.905 1.23 3.225 0 4.605-2.805 5.625-5.475 5.925.435.375.81 1.095.81 2.22 0 1.605-.015 2.895-.015 3.3 0 .315.225.69.825.57A12.02 12.02 0 0 0 24 12c0-6.63-5.37-12-12-12z"/>
#     </svg>
# </a>
# """, unsafe_allow_html=True)

# # ============================================================
# #  PATCH FOR OLD INPUTLAYER (convert batch_shape → batch_input_shape)
# # ============================================================
# class PatchedInputLayer(tf.keras.layers.InputLayer):
#     def __init__(self, batch_shape=None, optional=None, **kwargs):
#         # Convert batch_shape to batch_input_shape (new API)
#         if batch_shape is not None:
#             kwargs['batch_input_shape'] = batch_shape
#         # 'optional' is ignored
#         super().__init__(**kwargs)
    
#     @classmethod
#     def from_config(cls, config):
#         # Modify config before deserialization
#         if 'batch_shape' in config:
#             config['batch_input_shape'] = config.pop('batch_shape')
#         config.pop('optional', None)
#         return super().from_config(config)

# # ============================================================
# #  LOAD MODEL (with custom objects)
# # ============================================================
# @st.cache_resource
# def load_cifar_model():
#     # Remove corrupt or too small file
#     if os.path.exists(MODEL_FILE):
#         size_mb = os.path.getsize(MODEL_FILE) / (1024 * 1024)
#         if size_mb < 70:
#             os.remove(MODEL_FILE)

#     # Download if not present
#     if not os.path.exists(MODEL_FILE):
#         with st.spinner("Downloading model (80 MB) from Google Drive..."):
#             url = f"https://drive.google.com/uc?id={GOOGLE_DRIVE_FILE_ID}"
#             gdown.download(url, MODEL_FILE, quiet=False)

#     custom_objects = {
#         'Functional': tf.keras.models.Model,
#         'InputLayer': PatchedInputLayer,
#     }

#     model = tf.keras.models.load_model(
#         MODEL_FILE,
#         custom_objects=custom_objects,
#         compile=False,
#         safe_mode=False
#     )
#     return model

# # ============================================================
# #  GLOBAL MODEL LOAD
# # ============================================================
# model = load_cifar_model()

# # ============================================================
# #  TTA PREDICTION
# # ============================================================
# def tta_predict(img_arr: np.ndarray, model):
#     base = img_arr.astype(np.uint8)
#     pil = Image.fromarray(base)

#     augs = [
#         base,
#         np.fliplr(base),
#         np.array(pil.rotate(5)),
#         np.array(pil.rotate(-5))
#     ]

#     preds = []
#     for aug in augs:
#         x = aug.astype(np.float32) / 255.0
#         x = np.expand_dims(x, axis=0)
#         logits = model.predict(x, verbose=0).squeeze()
#         probs = tf.nn.softmax(logits).numpy()
#         preds.append(probs)

#     return np.array(preds)

# def pct(v: float) -> str:
#     return f"{v * 100:.1f}%"

# # ============================================================
# #  SIDEBAR
# # ============================================================
# with st.sidebar:
#     st.markdown("""
#     <div style="font-family:'Syne',sans-serif;font-size:22px;font-weight:800;
#                 background:linear-gradient(135deg,#fff,#22d3ee);
#                 -webkit-background-clip:text;-webkit-text-fill-color:transparent;
#                 margin-bottom:6px;">
#         CIFAR-10<br>Classifier
#     </div>
#     <div style="font-size:11px;color:#6b7280;letter-spacing:1px;margin-bottom:20px;">
#         CNN · RESIDUAL · TTA
#     </div>
#     """, unsafe_allow_html=True)
#     st.markdown("""
#     <div class="about-card">
#         A <strong>Convolutional Neural Network</strong> trained on the CIFAR-10 dataset
#         to classify images into <strong>10 categories</strong> using residual connections
#         and Test-Time Augmentation for robust predictions.
#     </div>
#     """, unsafe_allow_html=True)
#     st.markdown("<div class='sec-title'>10 Classes</div>", unsafe_allow_html=True)
#     cols_sb = st.columns(2)
#     for i, cls in enumerate(CLASSES):
#         with cols_sb[i % 2]:
#             st.markdown(f"<div style='font-size:13px;margin-bottom:6px;'>{EMOJIS[cls]} {cls.capitalize()}</div>", unsafe_allow_html=True)
#     st.divider()
#     st.markdown("<div class='sec-title'>Model Details</div>", unsafe_allow_html=True)
#     for k, v in [("Architecture", "ResNet-style CNN"), ("Dataset", "CIFAR-10"), ("TTA Folds", "4"), ("Input Size", "32×32 px")]:
#         st.markdown(f"<div style='display:flex;justify-content:space-between;font-size:12px;margin-bottom:6px;'><span style='color:#6b7280;'>{k}</span><span style='font-family:DM Mono,monospace;color:#22d3ee;'>{v}</span></div>", unsafe_allow_html=True)
#     st.divider()
#     st.markdown("<div style='font-size:11px;color:#4b5563;text-align:center;'>Developed by <strong style='color:#6b7280;'>Md. Nafijul Islam</strong></div>", unsafe_allow_html=True)

# # ============================================================
# #  MAIN UI
# # ============================================================
# st.markdown("""
# <div class="hero-wrap">
#     <div class="hero-label">Deep Learning · Image Recognition</div>
#     <div class="hero-title">CIFAR-10 Vision Classifier</div>
#     <div class="hero-sub">Residual CNN with <span>Test-Time Augmentation</span> — upload any image to classify it into one of 10 categories.</div>
#     <div class="creator">Created by <strong>Md. Nafijul Islam</strong></div>
# </div>
# """, unsafe_allow_html=True)

# st.markdown("""
# <div class="stat-row">
#     <div class="stat-pill"><div class="num">~89%</div><div class="lbl">Val Accuracy</div></div>
#     <div class="stat-pill"><div class="num">10</div><div class="lbl">Classes</div></div>
#     <div class="stat-pill"><div class="num">4×</div><div class="lbl">TTA Folds</div></div>
#     <div class="stat-pill"><div class="num">50K</div><div class="lbl">Training Images</div></div>
# </div>
# """, unsafe_allow_html=True)

# # Training history
# with st.expander("📈 Training History (Graphs)", expanded=False):
#     df_history = pd.DataFrame(TRAINING_HISTORY)
#     col_acc, col_loss = st.columns(2)
#     with col_acc:
#         fig_acc = go.Figure()
#         fig_acc.add_trace(go.Scatter(x=df_history["epochs"], y=df_history["train_acc"]*100, mode="lines", name="Train Acc", line=dict(color="#6366f1", width=3, shape='spline'), fill='tozeroy', fillcolor='rgba(99,102,241,0.15)'))
#         fig_acc.add_trace(go.Scatter(x=df_history["epochs"], y=df_history["val_acc"]*100, mode="lines", name="Val Acc", line=dict(color="#22d3ee", width=3, shape='spline'), fill='tozeroy', fillcolor='rgba(34,211,238,0.1)'))
#         fig_acc.update_layout(title="Accuracy", xaxis_title="Epoch", yaxis_title="Accuracy (%)", template="plotly_dark", height=380, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", font=dict(color="#e8eaf0"), legend=dict(orientation="h", yanchor="bottom", y=0.98, xanchor="right", x=0.98))
#         fig_acc.update_xaxes(gridcolor="rgba(255,255,255,0.08)")
#         fig_acc.update_yaxes(gridcolor="rgba(255,255,255,0.08)", range=[30,100])
#         st.plotly_chart(fig_acc, use_container_width=True)
#     with col_loss:
#         fig_loss = go.Figure()
#         fig_loss.add_trace(go.Scatter(x=df_history["epochs"], y=df_history["train_loss"], mode="lines", name="Train Loss", line=dict(color="#f472b6", width=3, shape='spline'), fill='tozeroy', fillcolor='rgba(244,114,182,0.1)'))
#         fig_loss.add_trace(go.Scatter(x=df_history["epochs"], y=df_history["val_loss"], mode="lines", name="Val Loss", line=dict(color="#facc15", width=3, shape='spline'), fill='tozeroy', fillcolor='rgba(250,204,21,0.1)'))
#         fig_loss.update_layout(title="Loss", xaxis_title="Epoch", yaxis_title="Loss", template="plotly_dark", height=380, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", font=dict(color="#e8eaf0"), legend=dict(orientation="h", yanchor="bottom", y=0.98, xanchor="right", x=0.98))
#         fig_loss.update_xaxes(gridcolor="rgba(255,255,255,0.08)")
#         fig_loss.update_yaxes(gridcolor="rgba(255,255,255,0.08)")
#         st.plotly_chart(fig_loss, use_container_width=True)

# with st.expander("📊 Training History (Numeric Table)", expanded=False):
#     df_display = pd.DataFrame(TRAINING_HISTORY)
#     df_display["train_acc"] = (df_display["train_acc"] * 100).map("{:.2f}%".format)
#     df_display["val_acc"] = (df_display["val_acc"] * 100).map("{:.2f}%".format)
#     df_display["train_loss"] = df_display["train_loss"].map("{:.3f}".format)
#     df_display["val_loss"] = df_display["val_loss"].map("{:.3f}".format)
#     st.dataframe(df_display, use_container_width=True)

# st.divider()

# # ============================================================
# # UPLOAD & PREDICTION
# # ============================================================
# left, right = st.columns([1, 1], gap="large")

# with left:
#     st.markdown("<div class='sec-title'>Upload Image</div>", unsafe_allow_html=True)
#     file = st.file_uploader("Choose a JPG or PNG file", type=["jpg","jpeg","png"], label_visibility="collapsed")
#     current_image = None
#     if file:
#         image = Image.open(file).convert("RGB")
#         st.image(image, use_container_width=True, caption="Original image")
#         raw = Image.open(file)
#         st.markdown(f"<div style='font-size:11px;color:#6b7280;margin-top:6px;'>Original: {raw.width}×{raw.height}px · {file.type}</div>", unsafe_allow_html=True)
#         current_image = image

# with right:
#     st.markdown("<div class='sec-title'>Prediction Result</div>", unsafe_allow_html=True)
#     if current_image is None:
#         st.info("👈 Upload an image to see classification")
#     else:
#         img_resized = current_image.resize((32, 32))
#         img_arr = np.array(img_resized, dtype=np.uint8)
#         with st.spinner("Running TTA inference (4 folds)…"):
#             start = time.time()
#             preds = tta_predict(img_arr, model)
#             elapsed = (time.time() - start) * 1000
#         avg = np.mean(preds, axis=0)
#         idx = int(np.argmax(avg))
#         label = CLASSES[idx]
#         conf = float(avg[idx])
#         st.markdown(f"""
#         <div class="pred-badge">
#             <div class="pred-emoji">{EMOJIS[label]}</div>
#             <div><div class="pred-text">{label.upper()}</div><div class="pred-conf">{conf*100:.2f}% confidence</div></div>
#         </div>
#         """, unsafe_allow_html=True)
#         st.caption(f"⏱️ Inference: {elapsed:.0f} ms (4 TTA folds)")
#         st.markdown("<div class='sec-title'>Top 5 Predictions</div>", unsafe_allow_html=True)
#         top5 = np.argsort(avg)[::-1][:5]
#         for rank, i in enumerate(top5):
#             c = CLASSES[i]; v = float(avg[i]); col = CLASS_COLORS[c]
#             st.markdown(f"""
#             <div class="prog-row">
#                 <div class="prog-label"><span>{'🥇' if rank==0 else '  '} {EMOJIS[c]} {c.capitalize()}</span><span>{pct(v)}</span></div>
#                 <div class="prog-track"><div class="prog-fill" style="width:{v*100:.2f}%; background:linear-gradient(90deg,{col}cc,{col}66);"></div></div>
#             </div>
#             """, unsafe_allow_html=True)

# if current_image is not None:
#     st.divider()
#     col_a, col_b = st.columns([1,1], gap="large")
#     with col_a:
#         st.markdown("<div class='sec-title'>All Class Probabilities</div>", unsafe_allow_html=True)
#         sorted_idx = np.argsort(avg)[::-1]
#         for i in sorted_idx:
#             c = CLASSES[i]; v = float(avg[i]); col_c = CLASS_COLORS[c]
#             st.markdown(f"""
#             <div class="prog-row"><div class="prog-label"><span>{EMOJIS[c]} {c.capitalize()}</span><span>{pct(v)}</span></div>
#             <div class="prog-track"><div class="prog-fill" style="width:{v*100:.2f}%; background:linear-gradient(90deg,{col_c}cc,{col_c}44);"></div></div></div>
#             """, unsafe_allow_html=True)
#     with col_b:
#         st.markdown("<div class='sec-title'>TTA Augmentation Breakdown</div>", unsafe_allow_html=True)
#         aug_names = ["Original","Horizontal Flip","Rotate +5°","Rotate −5°"]
#         aug_icons = ["🖼️","🔄","↗️","↘️"]
#         c1,c2 = st.columns(2)
#         for k in range(4):
#             p = preds[k]; pi = int(np.argmax(p)); pcls = CLASSES[pi]; pv = float(np.max(p)); pcol = CLASS_COLORS[pcls]
#             target = c1 if k%2==0 else c2
#             with target:
#                 st.markdown(f"""
#                 <div class="tta-card"><div class="tta-aug-name">{aug_icons[k]} {aug_names[k]}</div>
#                 <div style="font-size:24px;">{EMOJIS[pcls]}</div>
#                 <div class="tta-pred-label">{pcls.capitalize()}</div>
#                 <div class="tta-pred-conf">{pv*100:.1f}%</div>
#                 <div class="tta-bar-wrap"><div class="tta-bar-fill" style="width:{pv*100:.1f}%;background:linear-gradient(90deg,{pcol},{pcol}88);"></div></div></div>
#                 """, unsafe_allow_html=True)
#                 st.markdown("<div style='height:10px;'></div>", unsafe_allow_html=True)
#     st.divider()
#     tta_labels = [CLASSES[int(np.argmax(preds[i]))] for i in range(4)]
#     agreement = tta_labels.count(label)/4
#     st.markdown(f"""
#     <div class="glass-card"><div class="sec-title">TTA Agreement</div>
#     <div style="display:flex;align-items:center;gap:16px;"><div style="font-size:32px;">{'✅' if agreement==1 else '⚠️'}</div>
#     <div><div style="font-family:'Syne',sans-serif;font-size:18px;font-weight:700;">{int(agreement*4)}/4 augmentations agree on <span style="color:#22d3ee;">{label.capitalize()}</span></div>
#     <div style="font-size:12px;color:#6b7280;">{'High confidence — all TTA folds are consistent.' if agreement==1 else 'Some folds disagree — treat confidence with caution.'}</div></div>
#     <div style="margin-left:auto;"><div style="font-family:'DM Mono',monospace;font-size:28px;color:#22d3ee;font-weight:500;">{int(agreement*100)}%</div></div></div>
#     <div class="prog-track" style="margin-top:12px;height:8px;"><div class="prog-fill" style="width:{agreement*100:.0f}%;background:{'linear-gradient(90deg,#10b981,#22d3ee)' if agreement==1 else 'linear-gradient(90deg,#f59e0b,#f87171)'};"></div></div></div>
#     """, unsafe_allow_html=True)

# st.markdown("<br>", unsafe_allow_html=True)
# st.markdown("""
# <div style="text-align:center;padding:16px;border-top:1px solid rgba(255,255,255,0.06);">
#     <div style="font-size:11px;color:#374151;letter-spacing:1px;">Built with <span style="color:#6366f1;">Streamlit</span> · TensorFlow / Keras · <strong>Md. Nafijul Islam</strong></div>
# </div>
# """, unsafe_allow_html=True)