import streamlit as st
import numpy as np
import joblib
import cv2
from PIL import Image
import os
import hashlib
from datetime import datetime

import tensorflow as tf
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.applications.resnet50 import preprocess_input

from pymongo import MongoClient
import certifi

# ============================================================
# PAGE CONFIG (ANDROID FRIENDLY)
# ============================================================
st.set_page_config(
    page_title="Medicinal Plant Identification",
    layout="centered",
    initial_sidebar_state="expanded"
)

# ============================================================
# 🔹 UI ENHANCEMENT — GLOBAL THEME VARIABLES
# ============================================================
PRIMARY_COLOR = "#7CFF9B"
CARD_BG = "#121212"
BORDER_COLOR = "#2a2a2a"

# ============================================================
# GLOBAL UI STYLES (FIXED, NOT REMOVED)
# ============================================================
st.markdown(
    f"""
<style>
html, body, [class*="css"] {{
    background-color: #0e1117;
}}

.plant-card {{
    background-color: {CARD_BG};
    border-radius: 14px;
    padding: 18px 20px;
    border: 1px solid {BORDER_COLOR};
    box-shadow: 0 6px 14px rgba(0,0,0,0.45);
    color: #e6e6e6;
}}

.plant-title {{
    font-size: 1.35rem;
    font-weight: 600;
    margin-bottom: 10px;
    color: {PRIMARY_COLOR};
}}

.plant-row {{
    margin-bottom: 6px;
    font-size: 0.95rem;
    color: #dddddd;
}}

.plant-desc {{
    margin-top: 10px;
    font-size: 0.95rem;
    line-height: 1.5;
    color: #cccccc;
}}

.plant-card b {{
    color: #ffffff;
}}

.hero {{
    background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
    padding: 20px;
    border-radius: 16px;
    margin-bottom: 20px;
    border: 1px solid #2a2a2a;
}}

.hero h2 {{
    color: {PRIMARY_COLOR};
    margin-bottom: 6px;
}}

.hero p {{
    color: #d0d0d0;
    font-size: 0.95rem;
}}

.badge {{
    display: inline-block;
    padding: 4px 10px;
    border-radius: 10px;
    font-size: 0.75rem;
    margin-right: 6px;
    background-color: #1f1f1f;
    border: 1px solid #2a2a2a;
    color: #bbbbbb;
}}

[data-testid="metric-container"] {{
    background-color: #111;
    border: 1px solid {BORDER_COLOR};
    padding: 14px;
    border-radius: 12px;
}}

@media (max-width: 600px) {{
    .plant-card {{
        padding: 16px;
    }}
}}
</style>
""",
    unsafe_allow_html=True
)

# ============================================================
# HEADER
# ============================================================
st.markdown(
    """
<div class="hero">
    <h2>🌿 Medicinal Plant Identification</h2>
    <p>
        AI-powered recognition using <b>ResNet50 + QPSO + SVM</b> with explainable
        Grad-CAM visual insights and Firebase logging.
    </p>
    <span class="badge">Model: ResNet50 + QPSO + SVM</span>
    <span class="badge">Explainable AI</span>
</div>
""",
    unsafe_allow_html=True
)

st.divider()
# ============================================================
# SIDEBAR – PRODUCTION CONTROLS
# ============================================================
st.sidebar.markdown("## ⚙️ System Controls")

CONF_THRESHOLD = st.sidebar.slider(
    "Prediction Confidence Threshold",
    min_value=0.50,
    max_value=0.90,
    value=0.60,
    step=0.05
)

upload_to_firestore = st.sidebar.checkbox(
    "Enable Firestore Logging",
    value=True
)

st.sidebar.markdown("---")
st.sidebar.markdown(f"### 🔒 Active Threshold: `{CONF_THRESHOLD}`")

# 🔹 UI ENHANCEMENT (ADDED) — CONFIDENCE EXPLANATION
st.sidebar.info(
    "Predictions below this confidence will be marked as **Unknown**.\n\n"
    "Recommended: **0.60 – 0.70** for real-world usage."
)

# ============================================================
# DISPLAY CONFIG
# ============================================================
DISPLAY_IMG_SIZE = (500, 500)
DISPLAY_GRADCAM_SIZE = (500, 500)
IMG_SIZE = (224, 224)

# ============================================================
# PATH CONFIG
# ============================================================
PROJECT_ROOT = os.getcwd()

RESNET_MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "resnet_finetuned_model_tf.keras")
MODEL_DIR = os.path.join(PROJECT_ROOT, "models_finetuned")
QPSO_DIR = os.path.join(PROJECT_ROOT, "qpso_finetuned")

SVM_MODEL_PATH = os.path.join(MODEL_DIR, "qpso_svm_model_finetuned.pkl")
SCALER_PATH = os.path.join(MODEL_DIR, "qpso_scaler_finetuned.pkl")
QPSO_MASK_PATH = os.path.join(QPSO_DIR, "qpso_selected_mask_finetuned.npy")
CLASS_NAMES_PATH = os.path.join(PROJECT_ROOT, "class_names.npy")

# ============================================================
# LOAD MODELS
# ============================================================
@st.cache_resource
def load_assets():
    resnet_model = load_model(RESNET_MODEL_PATH)

    from tensorflow.keras.layers import GlobalAveragePooling2D, GlobalMaxPool2D
    for layer in reversed(resnet_model.layers):
        if isinstance(layer, (GlobalAveragePooling2D, GlobalMaxPool2D)):
            gap_output = layer.output
            break
    else:
        gap_output = resnet_model.layers[-2].output

    feature_extractor = Model(resnet_model.input, gap_output)

    svm_model = joblib.load(SVM_MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)

    mask = np.load(QPSO_MASK_PATH)
    selected_indices = np.where(mask > 0.5)[0]

    class_names = np.load(CLASS_NAMES_PATH, allow_pickle=True).tolist()

    return resnet_model, feature_extractor, svm_model, scaler, selected_indices, class_names


resnet_model, feature_extractor, svm_model, scaler, selected_indices, CLASS_NAMES = load_assets()

# ============================================================
# MONGODB
# ============================================================
MONGO_URI = "mongodb+srv://medical:d12345@plantdata.8mmiqsk.mongodb.net/"
client = MongoClient(MONGO_URI, tlsCAFile=certifi.where())
db = client["plant_db"]
collection = db["plants"]

def get_mongo_details(class_name):
    doc = collection.find_one(
        {"plant_name": {"$regex": f"^{class_name}$", "$options": "i"}}
    )
    if not doc:
        return {
            "local_name": "N/A",
            "scientific_name": class_name,
            "genus": "N/A",
            "family": "N/A",
            "description": "No description available."
        }
    return doc

# ============================================================
# IMAGE QUALITY CHECK
# ============================================================
def check_image_quality(pil_img):
    img = np.array(pil_img)

    if img.shape[0] < 200 or img.shape[1] < 200:
        return False, "Low resolution image."

    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    if cv2.Laplacian(gray, cv2.CV_64F).var() < 80:
        return False, "Image appears blurred."

    edges = cv2.Canny(gray, 100, 200)
    if np.sum(edges > 0) / edges.size < 0.02:
        return False, "Leaf not clearly visible."

    return True, None

# ============================================================
# GRAD-CAM
# ============================================================
def find_last_conv_layer(model):
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer.name
    raise ValueError("No Conv2D layer found")

def compute_gradcam(model, img_array, layer_name):
    grad_model = Model(
        model.inputs,
        [model.get_layer(layer_name).output, model.output]
    )
    with tf.GradientTape() as tape:
        conv_out, preds = grad_model({"input_layer": img_array})
        loss = tf.reduce_max(preds)

    grads = tape.gradient(loss, conv_out)
    pooled = tf.reduce_mean(grads, axis=(0, 1, 2))

    heatmap = conv_out[0] @ pooled[..., tf.newaxis]
    heatmap = tf.maximum(tf.squeeze(heatmap), 0)
    heatmap /= tf.reduce_max(heatmap)

    return heatmap.numpy()

def overlay_gradcam(pil_img, heatmap, alpha=0.4):
    img = np.array(pil_img)
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
    return cv2.addWeighted(img, 1 - alpha, heatmap, alpha, 0)

# ============================================================
# 🔹 ADDITION AS REQUESTED (BEFORE FIRESTORE LOGGING)
# ============================================================
import base64
import io

def image_to_base64(pil_image):
    buffer = io.BytesIO()
    pil_image.save(buffer, format="JPEG")
    encoded = base64.b64encode(buffer.getvalue()).decode()
    return f"data:image/jpeg;base64,{encoded}"

# ============================================================
# MAIN UI
# ============================================================
uploaded = st.file_uploader("📷 Upload a leaf image", type=["jpg", "jpeg", "png"])

if uploaded:
    uploaded.seek(0)
    image = Image.open(uploaded).convert("RGB")

    # 🔹 UI ENHANCEMENT (ADDED) — CENTERED IMAGE CARD
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.image(image.resize(DISPLAY_IMG_SIZE), caption="Uploaded Leaf Image")

    ok, msg = check_image_quality(image)
    if not ok:
        st.error(msg)
        st.stop()

    img_arr = preprocess_input(
        np.expand_dims(np.array(image.resize(IMG_SIZE)), axis=0)
    )

    features = feature_extractor.predict({"input_layer": img_arr})
    features = scaler.transform(features[:, selected_indices])

    probs = svm_model.predict_proba(features)[0]
    top3_idx = np.argsort(probs)[::-1][:3]
    top3 = [(CLASS_NAMES[i], probs[i]) for i in top3_idx]

    st.divider()
    st.markdown("## 🔍 Prediction Results")

    c1, c2, c3 = st.columns(3)
    for col, (cls, p) in zip([c1, c2, c3], top3):
        col.metric(label=cls, value=f"{p:.2f}")

    max_conf = top3[0][1]

    # 🔹 UI ENHANCEMENT (ADDED) — CONFIDENCE BAR
    st.progress(min(max_conf, 1.0))

    if max_conf < CONF_THRESHOLD:
        st.warning("Prediction confidence is low. Result marked as **Unknown**.")
    else:
        predicted_class = top3[0][0]
        st.success(f"Final Prediction: **{predicted_class}**")

        plant = get_mongo_details(predicted_class)

        st.markdown("## 🌿 Plant Information")
        st.markdown(
            f"""
<div class="plant-card">
    <div class="plant-title">{plant.get("local_name", "N/A")}</div>
    <div class="plant-row"><b>Scientific Name:</b> <i>{plant.get("scientific_name", "N/A")}</i></div>
    <div class="plant-row"><b>Genus:</b> {plant.get("genus", "N/A")}</div>
    <div class="plant-row"><b>Family:</b> {plant.get("family", "N/A")}</div>
    <div class="plant-desc"><b>Description:</b><br>{plant.get("description", "No description available.")}</div>
</div>
""",
            unsafe_allow_html=True
        )

        st.markdown("## 🔥 Explainability (Grad-CAM)")
        heatmap = compute_gradcam(
            resnet_model,
            img_arr,
            find_last_conv_layer(resnet_model)
        )
        gradcam = overlay_gradcam(image, heatmap)

        c1, c2, c3 = st.columns([1, 2, 1])
        with c2:
            st.image(
                cv2.resize(gradcam, DISPLAY_GRADCAM_SIZE),
                caption="Grad-CAM Visualization"
            )

# ============================================================
# 🔥 FIREBASE SECTION (UNCHANGED)
# ============================================================
import firebase_admin
from firebase_admin import credentials, firestore

@st.cache_resource
def init_firestore():
    if not firebase_admin._apps:
        cred = credentials.Certificate(dict(st.secrets["firebase"]))
        firebase_admin.initialize_app(cred)
    return firestore.client()

firestore_db = init_firestore()

def log_to_firestore(payload: dict):
    if upload_to_firestore:
        try:
            firestore_db.collection("prediction_logs").add(payload)
            st.success("✅ Prediction successfully stored in Firebase")
        except Exception as e:
            st.error(f"❌ Firebase logging failed: {e}")

if uploaded and max_conf >= CONF_THRESHOLD:
    log_to_firestore({
        "SVM Class": predicted_class,
        "SVM Confidence": float(max_conf),
        "Scientific Name": plant.get("scientific_name", "N/A"),
        "Genus": plant.get("genus", "N/A"),
        "Family": plant.get("family", "N/A"),
        "Wikipedia Summary": plant.get("description", ""),
        "Image URL": image_to_base64(image),
        "Timestamp": datetime.utcnow().isoformat(),
        "Top3": {cls: float(p) for cls, p in top3},
        "Threshold": CONF_THRESHOLD,
        "Model": "ResNet50 + QPSO + SVM"
    })

# ============================================================
# FOOTER
# ============================================================
st.divider()
st.caption("© Team LeafLogic | Production-Ready AI System for Medicinal Plant Identification")
