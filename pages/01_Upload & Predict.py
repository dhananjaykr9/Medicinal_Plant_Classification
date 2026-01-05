import streamlit as st
import numpy as np
import joblib
import cv2
from PIL import Image
import os
import hashlib
from datetime import datetime
import base64

import tensorflow as tf
#from tensorflow.keras.models import load_model, Model
#from tensorflow.keras.applications.resnet50 import preprocess_input

import keras
from keras.models import load_model, Model
from keras.applications.resnet50 import preprocess_input


from pymongo import MongoClient
import certifi
from zoneinfo import ZoneInfo
import shap
import matplotlib.pyplot as plt

# Import the agent
from agent import PredictionAgent

# ============================================================
# IST TIME HELPER (CORRECT)
# ============================================================
def now_ist():
    return datetime.now(ZoneInfo("Asia/Kolkata"))

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="Medicinal Plant Identification",
    layout="centered",
    initial_sidebar_state="expanded"
)

# ============================================================
# BASE64 HELPERS (BOTH KEPT — NO REMOVAL)
# ============================================================
def load_base64(path):
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()

def load_base64_image(path):
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()

# ============================================================
# LOAD LOGO
# ============================================================
LOGO_PATH = "assets/logo_leaflogic.png"
logo_base64 = load_base64(LOGO_PATH)

if logo_base64 is None:
    st.error("❌ Logo not found in assets/logo_leaflogic.png")

# ============================================================
# SESSION STATE
# ============================================================
if "logged_doc_id" not in st.session_state:
    st.session_state.logged_doc_id = None

# ============================================================
# UI THEME VARIABLES
# ============================================================
PRIMARY_COLOR = "#7CFF9B"
CARD_BG = "#121212"
BORDER_COLOR = "#2a2a2a"

# ============================================================
# GLOBAL UI STYLES (FIXED)
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

.card {{
    background-color: #1c1c1c;
    padding: 1.2rem;
    border-radius: 12px;
    box-shadow: 0 4px 15px rgba(0,0,0,0.4);
    margin-bottom: 1rem;
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

/* ================= SPLASH SCREEN ================= */
#splash-screen {{
    position: fixed;
    inset: 0;
    background: radial-gradient(circle at top, #0f2027, #0e1117 60%);
    display: flex;
    align-items: center;
    justify-content: center;
    z-index: 9999;
    animation: hideSplash 0.6s ease-out forwards;
    animation-delay: 2.6s;
}}

.splash-content {{
    text-align: center;
}}

.splash-logo {{
    width: 240px;
    max-width: 70vw;
    margin-bottom: 20px;
    animation: logoPop 1.4s ease-out forwards;
}}

.splash-title {{
    font-size: 2.3rem;
    font-weight: 700;
    color: #90ee90;
    opacity: 0;
    animation: fadeUp 1s ease-out forwards;
    animation-delay: 0.9s;
}}

.splash-sub {{
    font-size: 1.15rem;
    color: #cfcfcf;
    opacity: 0;
    animation: fadeUp 1s ease-out forwards;
    animation-delay: 1.3s;
}}

@keyframes logoPop {{
    0% {{ opacity: 0; transform: scale(0.5); }}
    60% {{ opacity: 1; transform: scale(1.08); }}
    100% {{ transform: scale(1); }}
}}

@keyframes fadeUp {{
    0% {{ opacity: 0; transform: translateY(18px); }}
    100% {{ opacity: 1; transform: translateY(0); }}
}}

@keyframes hideSplash {{
    to {{ opacity: 0; visibility: hidden; }}
}}
</style>
""",
    unsafe_allow_html=True
)

# ============================================================
# SPLASH SCREEN
# ============================================================
if logo_base64:
    st.markdown(
        f"""
<div id="splash-screen">
    <div class="splash-content">
        <img src="data:image/png;base64,{logo_base64}" class="splash-logo">
        <div class="splash-title">Medicinal Plant Identifier</div>
        <div class="splash-sub">AI-Powered Plant Detection System</div>
    </div>
</div>
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

st.markdown("""
<div class="card" style="border-left: 5px solid #ff9800;">
<b>⚠️ Disclaimer Known Classes Only</b><br>
This system is trained on <b>five</b> selected medicinal plant species: <b>Neem, Tulsi, Aloe Vera, Moringa, Hibiscus</b>. Predictions are <b>strictly restricted</b> to these classes only.
</div>
""", unsafe_allow_html=True)

#st.warning("We do not claim 100% Accuracy")

st.markdown(
    """
<div class="card" style="border-left: 5px solid #ff9800;">
<b>Important Notice:</b><br>
This system does <b>not claim 100% accuracy</b>.  
Predictions are generated using AI models trained on limited datasets and may vary based on image quality and environmental conditions
and intended for <b>educational and research purposes only</b>.
</div>
""",
    unsafe_allow_html=True
)


# ============================================================
# SAFE DEFAULT STATE VARIABLES (CRITICAL FIX)
# ============================================================
predicted_class = None
max_conf = None
plant = None
top3 = None

if "feedback_submitted" not in st.session_state:
    st.session_state.feedback_submitted = False

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

    from keras.layers import GlobalAveragePooling2D, GlobalMaxPooling2D
    for layer in reversed(resnet_model.layers):
        if isinstance(layer, (GlobalAveragePooling2D, GlobalMaxPooling2D)):
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
# SHAP BACKGROUND (FAST & SAFE)
# ============================================================
@st.cache_resource
def load_shap_background():
    # Use small random background for speed
    bg = np.random.normal(
        size=(30, len(selected_indices))
    )
    return bg

shap_background = load_shap_background()


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
image_hash = None

uploaded = st.file_uploader("📷 Upload a leaf image", type=["jpg", "jpeg", "png"])

if uploaded:
    uploaded.seek(0)
    image = Image.open(uploaded).convert("RGB")

    # ============================================================
    # 🔑 UNIQUE IMAGE IDENTIFIER (CRITICAL FIX)
    # ============================================================
    image_bytes = uploaded.getvalue()
    image_hash = hashlib.md5(image_bytes).hexdigest()

    # 🔄 Reset feedback state ONLY when a new image is uploaded
    if st.session_state.get("last_image_hash") != image_hash:
        st.session_state.feedback_submitted = False
        st.session_state.user_feedback = None
        st.session_state.feedback_logged = False
        st.session_state.logged_doc_id = None
        st.session_state.last_image_hash = image_hash

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

    # ============================================================
    # 🤖 AGENTIC AI DECISION (NEW INTEGRATION)
    # ============================================================
    agent = PredictionAgent()
    predicted_class = top3[0][0]
    pred_confidence = max_conf
    
    agent_response = agent.decide(confidence=pred_confidence)
    if agent_response["decision"] == "REUPLOAD":
        st.warning(agent_response["message"])
        st.stop()
    #st.write("Agent decision:", agent_response["decision"])

    # 🔹 UI ENHANCEMENT (ADDED) — CONFIDENCE BAR
    st.progress(min(max_conf, 1.0))

    if max_conf < CONF_THRESHOLD:
        st.warning("Prediction confidence is low. Result marked as **Unknown**. This plant may be outside the trained medicinal plant classes.")
    else:
        # predicted_class already assigned for the agent above
        st.success(f"Final Prediction: **{predicted_class}**")

        plant = get_mongo_details(predicted_class)
        st.info("Confidence represents relative likelihood among trained classes, not **absolute certainty**.")
                
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

        def svm_predict_proba_wrapper(X_reduced):
            """
            X_reduced: (n_samples, TOP_K)
            Returns: predict_proba on full feature space
            """
            X_full = np.zeros((X_reduced.shape[0], features.shape[1]))
            X_full[:, top_k_idx] = X_reduced
            return svm_model.predict_proba(X_full)

        # ============================================================
        # 🔎 EXPLAINABILITY — SHAP (FAST MODE)
        # ============================================================
        st.markdown("## 🔎 Explainability (SHAP – Feature Level)")

        TOP_K = 30   # 🔥 change to 50 if you want slightly more detail

        # Approximate feature importance (distance-based proxy)
        importance = np.abs(features[0])

        top_k_idx = np.argsort(importance)[-TOP_K:]

        features_shap = features[:, top_k_idx]
        background_shap = shap_background[:, top_k_idx]

        with st.spinner("Computing SHAP explanation (optimized)..."):
            explainer = shap.Explainer(
                svm_predict_proba_wrapper,
                background_shap,
                algorithm="permutation",
                max_evals=2 * TOP_K + 1
            )

            shap_values = explainer(features_shap)
            
        pred_class_idx = CLASS_NAMES.index(predicted_class)
        shap_vals_class = shap_values.values[0, :, pred_class_idx]

        top_idx = np.argsort(np.abs(shap_vals_class))[-10:][::-1]

        fig, ax = plt.subplots(figsize=(8, 4))

        ax.barh(
            range(len(top_idx)),
            shap_vals_class[top_idx],
            color="#90ee90"
        )

        ax.set_yticks(range(len(top_idx)))
        ax.set_yticklabels([f"Feature {top_k_idx[i]}" for i in top_idx])
        ax.set_xlabel("SHAP Contribution")
        ax.set_title("Top Feature Contributions (SHAP)")
        ax.invert_yaxis()

        st.pyplot(fig)
        plt.close(fig)


        # ============================================================
        # 🔄 USER FEEDBACK — HUMAN IN THE LOOP
        # ============================================================
        st.markdown("## 🧠 Was this prediction correct?")

        st.info(
            "Your feedback helps us improve the system.\n\n"
            "**This is for future dataset expansion and model improvement.**"
        )

        if not st.session_state.feedback_submitted:

            col1, col2 = st.columns(2)

            with col1:
                if st.button("✅ Yes, Correct"):
                    st.session_state.user_feedback = "yes"
                    st.session_state.feedback_submitted = True
                    st.rerun()

            with col2:
                if st.button("❌ No, Incorrect"):
                    st.session_state.user_feedback = "no"
                    st.session_state.feedback_submitted = True
                    st.rerun()

        else:
            st.success("🙏 Thank you! Your feedback was recorded.")

# ============================================================
# 🔥 FIREBASE SECTION
# ============================================================
if "logged_image_hash" not in st.session_state:
    st.session_state.logged_image_hash = None

import firebase_admin
from firebase_admin import credentials, firestore

@st.cache_resource
def init_firestore():
    if not firebase_admin._apps:
        cred = credentials.Certificate(dict(st.secrets["firebase"]))
        firebase_admin.initialize_app(cred)
    return firestore.client()

firestore_db = init_firestore()

if "user_feedback" not in st.session_state:
    st.session_state.user_feedback = None

def log_to_firestore(payload: dict):
    if upload_to_firestore:
        try:
            doc_ref = firestore_db.collection("prediction_logs").add(payload)
            st.success("✅ Prediction successfully stored in Firebase")
            return doc_ref[1].id   # 🔑 return document ID
        except Exception as e:
            st.error(f"❌ Firebase logging failed: {e}")
    return None

# ============================================================
# 🔐 LOG ONLY ONCE PER IMAGE (CRITICAL FIX)
# ============================================================
if (
    uploaded
    and max_conf is not None
    and max_conf >= CONF_THRESHOLD
    and st.session_state.logged_image_hash != image_hash
):

    combined_payload = {
        "SVM Class": predicted_class,
        "SVM Confidence": float(max_conf),
        "Scientific Name": plant.get("scientific_name", "N/A"),
        "Genus": plant.get("genus", "N/A"),
        "Family": plant.get("family", "N/A"),
        "Wikipedia Summary": plant.get("description", ""),
        "Image URL": image_to_base64(image),
        "Top3": {cls: float(p) for cls, p in top3},
        "Threshold": CONF_THRESHOLD,
        "Model": "ResNet50 + QPSO + SVM",
        "Timestamp": now_ist().isoformat(),  
        
        # ❗ feedback initially empty
        "User Feedback": "Not Provided",
        "Feedback Purpose": "Future dataset expansion"
    }

    doc_id = log_to_firestore(combined_payload)
    st.session_state.logged_image_hash = image_hash
    st.session_state.logged_doc_id = doc_id

if "feedback_logged" not in st.session_state:
    st.session_state.feedback_logged = False

if (
    st.session_state.feedback_submitted
    and st.session_state.user_feedback is not None
    and st.session_state.logged_doc_id is not None
    and not st.session_state.feedback_logged
):
    firestore_db.collection("prediction_logs") \
        .document(st.session_state.logged_doc_id) \
        .update({
            "User Feedback": st.session_state.user_feedback,
            "Feedback Timestamp": now_ist().isoformat()

        })

    st.session_state.feedback_logged = True

# ============================================================
# FOOTER
# ============================================================
st.divider()
st.caption("© Team LeafLogic | AI System for Medicinal Plant Identification")
