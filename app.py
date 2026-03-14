import torch
import torchvision.models as models
from utils.preprocess import preprocess
import streamlit as st
from PIL import Image
import time
import random  # Replace with real model inference

# ─── Page Config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="WasteAI — Smart Classification",
    page_icon="♻️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600&display=swap');

/* Base */
html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: #0d0d0d;
    color: #e8e8e0;
}

/* Header */
.main-header {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 2rem 0 0.5rem;
    border-bottom: 2px solid #2a2a2a;
    margin-bottom: 2rem;
}
.main-title {
    font-family: 'Space Mono', monospace;
    font-size: 2.2rem;
    font-weight: 700;
    letter-spacing: -1px;
    color: #b5ff47;
    margin: 0;
}
.main-subtitle {
    font-size: 0.85rem;
    color: #666;
    letter-spacing: 2px;
    text-transform: uppercase;
    margin-top: 4px;
}

/* Metric cards */
.metric-card {
    background: #161616;
    border: 1px solid #2a2a2a;
    border-radius: 8px;
    padding: 1.2rem 1.5rem;
    margin-bottom: 1rem;
}
.metric-card h4 {
    font-family: 'Space Mono', monospace;
    font-size: 0.7rem;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: #555;
    margin: 0 0 8px 0;
}
.metric-label {
    font-size: 1.4rem;
    font-weight: 600;
    color: #e8e8e0;
}
.metric-badge {
    display: inline-block;
    padding: 2px 10px;
    border-radius: 3px;
    font-family: 'Space Mono', monospace;
    font-size: 0.8rem;
}
.badge-cnn  { background: #1a2e00; color: #b5ff47; border: 1px solid #b5ff47; }
.badge-clip { background: #001a2e; color: #47c8ff; border: 1px solid #47c8ff; }

/* Confidence bar */
.conf-bar-wrap { margin-top: 10px; }
.conf-label { font-size: 0.78rem; color: #888; margin-bottom: 4px; }
.conf-bar-outer {
    background: #222;
    border-radius: 2px;
    height: 6px;
    width: 100%;
}
.conf-bar-inner {
    height: 6px;
    border-radius: 2px;
    transition: width 0.6s ease;
}
.bar-cnn  { background: #b5ff47; }
.bar-clip { background: #47c8ff; }

/* Category tags */
.tag {
    display: inline-block;
    background: #1e1e1e;
    border: 1px solid #333;
    border-radius: 3px;
    padding: 3px 10px;
    font-size: 0.78rem;
    color: #aaa;
    margin: 3px 3px 3px 0;
    font-family: 'Space Mono', monospace;
}

/* Upload zone */
.upload-hint {
    font-size: 0.8rem;
    color: #555;
    text-align: center;
    padding: 8px 0;
    letter-spacing: 1px;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: #111 !important;
    border-right: 1px solid #222;
}
.sidebar-stat {
    background: #181818;
    border-left: 3px solid #b5ff47;
    padding: 8px 12px;
    margin-bottom: 8px;
    border-radius: 0 4px 4px 0;
    font-size: 0.85rem;
}
.sidebar-stat span { color: #b5ff47; font-weight: 600; }

/* Status pill */
.status-pill {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: #1a1a1a;
    border: 1px solid #333;
    border-radius: 20px;
    padding: 4px 12px;
    font-size: 0.78rem;
    color: #aaa;
    margin-bottom: 1.5rem;
}
.dot { width: 7px; height: 7px; border-radius: 50%; background: #b5ff47; animation: pulse 2s infinite; }
@keyframes pulse { 0%,100%{opacity:1} 50%{opacity:0.3} }

/* Divider */
hr { border-color: #222 !important; }
</style>
""", unsafe_allow_html=True)


# ─── Fake Model Inference (Replace with real models) ─────────────────────────
WASTE_CATEGORIES = {
    "Plastic": {"recyclable": True,  "color": "#b5ff47", "icon": "🧴"},
    "Paper":   {"recyclable": True,  "color": "#ffd447", "icon": "📄"},
    "Glass":   {"recyclable": True,  "color": "#47c8ff", "icon": "🍾"},
    "Metal":   {"recyclable": True,  "color": "#ff9f47", "icon": "🥫"},
    "Organic": {"recyclable": False, "color": "#8bff8b", "icon": "🍂"},
    "E-waste": {"recyclable": False, "color": "#ff6b6b", "icon": "📱"},
    "Other":   {"recyclable": False, "color": "#888",    "icon": "🗑️"},
}

SAMPLE_IMAGES = {
    "🧴 Plastic Bottle": {"cnn": ("Plastic", 0.94), "clip": ("Plastic", 0.91)},
    "📄 Paper":          {"cnn": ("Paper",   0.88), "clip": ("Paper",   0.85)},
    "🍾 Glass Bottle":   {"cnn": ("Glass",   0.91), "clip": ("Glass",   0.87)},
    "🥫 Metal Can":      {"cnn": ("Metal",   0.89), "clip": ("Metal",   0.93)},
    "🍂 Organic Waste":  {"cnn": ("Organic", 0.78), "clip": ("Organic", 0.82)},
}

def run_cnn_model(image: Image.Image):
    """Replace this stub with your real CNN inference."""
    time.sleep(0.4)
    cat = random.choice(list(WASTE_CATEGORIES.keys()))
    conf = round(random.uniform(0.75, 0.98), 2)
    return cat, conf

def run_clip_model(image: Image.Image):
    """Replace this stub with your real CLIP inference."""
    time.sleep(0.3)
    cat = random.choice(list(WASTE_CATEGORIES.keys()))
    conf = round(random.uniform(0.70, 0.97), 2)
    return cat, conf

def render_prediction_card(model_name: str, category: str, confidence: float, badge_class: str, bar_class: str):
    info = WASTE_CATEGORIES.get(category, {"recyclable": False, "color": "#888", "icon": "🗑️"})
    recyclable_str = "✓ Recyclable" if info["recyclable"] else "✗ Non-recyclable"
    pct = int(confidence * 100)
    st.markdown(f"""
    <div class="metric-card">
        <h4>{model_name} &nbsp;<span class="metric-badge {badge_class}">{model_name}</span></h4>
        <div class="metric-label">{info['icon']} {category}</div>
        <div style="font-size:0.8rem; color:{'#b5ff47' if info['recyclable'] else '#ff6b6b'}; margin-top:4px;">
            {recyclable_str}
        </div>
        <div class="conf-bar-wrap">
            <div class="conf-label">Confidence — {pct}%</div>
            <div class="conf-bar-outer">
                <div class="conf-bar-inner {bar_class}" style="width:{pct}%"></div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ─── Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ♻️ WasteAI")
    st.markdown("<hr>", unsafe_allow_html=True)

    st.markdown("**Models Active**")
    st.markdown('<div class="sidebar-stat">CNN &nbsp;<span>ResNet-50</span></div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-stat">CLIP &nbsp;<span>ViT-B/32</span></div>', unsafe_allow_html=True)

    st.markdown("<br>**Supported Categories**", unsafe_allow_html=True)
    for cat, meta in WASTE_CATEGORIES.items():
        st.markdown(f"<span class='tag'>{meta['icon']} {cat}</span>", unsafe_allow_html=True)

    st.markdown("<br><hr>", unsafe_allow_html=True)
    st.markdown("**Session Stats**")
    if "count" not in st.session_state:
        st.session_state.count = 0
    st.markdown(f"Images classified: **{st.session_state.count}**")

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown(
        '<div style="font-size:0.75rem;color:#444;">Replace stub functions with real model '
        'inference to deploy.</div>',
        unsafe_allow_html=True,
    )


# ─── Main Content ─────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <div>
        <p class="main-title">WASTE//AI</p>
        <p class="main-subtitle">CNN vs CLIP — Dual-Model Classification System</p>
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown('<div class="status-pill"><div class="dot"></div>Models ready</div>', unsafe_allow_html=True)

# ─── Upload Section ───────────────────────────────────────────────────────────
uploaded_file = st.file_uploader(
    "Upload a waste image",
    type=["jpg", "jpeg", "png", "webp"],
    help="Supported: JPG, PNG, WEBP",
    label_visibility="collapsed",
)
st.markdown('<p class="upload-hint">↑ DROP AN IMAGE OR CLICK TO BROWSE — JPG / PNG / WEBP</p>', unsafe_allow_html=True)

# ─── Sample Image Picker ──────────────────────────────────────────────────────
st.markdown("**Or try a sample:**")
sample_cols = st.columns(len(SAMPLE_IMAGES))
selected_sample = None
for col, (label, preds) in zip(sample_cols, SAMPLE_IMAGES.items()):
    with col:
        if st.button(label, use_container_width=True):
            selected_sample = (label, preds)

st.markdown("<hr>", unsafe_allow_html=True)

# ─── Run Classification ───────────────────────────────────────────────────────
image_to_classify = None
results = None

if uploaded_file is not None:
    image_to_classify = Image.open(uploaded_file).convert("RGB")
    with st.spinner("Running inference…"):
        cnn_cat, cnn_conf   = run_cnn_model(image_to_classify)
        clip_cat, clip_conf = run_clip_model(image_to_classify)
    results = {"cnn": (cnn_cat, cnn_conf), "clip": (clip_cat, clip_conf)}
    st.session_state.count += 1

elif selected_sample:
    label, preds = selected_sample
    results = preds  # pre-baked demo results

if image_to_classify or selected_sample:
    left, right = st.columns([1, 2], gap="large")

    with left:
        st.markdown("**Input Image**")
        if image_to_classify:
            st.image(image_to_classify, use_column_width=True)
        else:
            st.info(f"Sample: {selected_sample[0]}")

    with right:
        st.markdown("**Model Predictions**")
        c1, c2 = st.columns(2, gap="medium")
        with c1:
            cnn_cat, cnn_conf = results["cnn"]
            render_prediction_card("CNN", cnn_cat, cnn_conf, "badge-cnn", "bar-cnn")
        with c2:
            clip_cat, clip_conf = results["clip"]
            render_prediction_card("CLIP", clip_cat, clip_conf, "badge-clip", "bar-clip")

        # Agreement indicator
        agree = cnn_cat == clip_cat
        st.markdown(
            f"<div style='margin-top:0.5rem; font-size:0.82rem; color:{'#b5ff47' if agree else '#ff9f47'};'>"
            f"{'✓ Models agree' if agree else '⚠ Models disagree — consider manual review'}</div>",
            unsafe_allow_html=True,
        )

        # Disposal tip
        info = WASTE_CATEGORIES.get(cnn_cat, {})
        if info:
            rec = info.get("recyclable", False)
            st.markdown(
                f"<div class='metric-card' style='margin-top:1rem;'>"
                f"<h4>Disposal Guidance</h4>"
                f"<div style='font-size:0.9rem;'>{'Place in the <b>recycling bin</b>. Clean before disposing.' if rec else 'Place in <b>general waste</b>. Check local rules for special disposal.'}</div>"
                f"</div>",
                unsafe_allow_html=True,
            )
model = models.resnet18()
model.fc = torch.nn.Linear(model.fc.in_features, 4)
model.eval()
def predict_cnn(image):

    tensor = preprocess(image)

    outputs = model(tensor)

    _, predicted = torch.max(outputs, 1)

    classes = ["plastic","paper","glass","metal"]

    return classes[predicted.item()]
from transformers import CLIPProcessor, CLIPModel

clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
def predict_clip(image):

    labels = ["plastic waste","paper waste","glass waste","metal waste"]

    inputs = processor(
        text=labels,
        images=image,
        return_tensors="pt",
        padding=True
    )

    outputs = clip_model(**inputs)

    probs = outputs.logits_per_image.softmax(dim=1)

    index = probs.argmax()

    return labels[index]
cnn_result = predict_cnn(image)
clip_result = predict_clip(image)

st.success(f"CNN Prediction: {cnn_result}")
st.info(f"CLIP Prediction: {clip_result}")