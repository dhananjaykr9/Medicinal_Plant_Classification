import streamlit as st
import base64
import os
from agent import PredictionAgent


# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="Medicinal Plant Identifier",
    page_icon="🌿",
    layout="centered",
    initial_sidebar_state="expanded"
)

# ============================================================
# HELPER — BASE64 IMAGE LOADER (CRITICAL FIX)
# ============================================================
def load_base64(path):
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()

# ============================================================
# 🔹 ADDITION (FIX) — SECOND HELPER USED LATER
# ============================================================
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
# CSS — SPLASH + MAIN UI
# ============================================================
st.markdown("""
<style>
/* ================= SPLASH SCREEN ================= */
#splash-screen {
    position: fixed;
    inset: 0;
    background: radial-gradient(circle at top, #0f2027, #0e1117 60%);
    display: flex;
    align-items: center;
    justify-content: center;
    z-index: 9999;
    animation: hideSplash 0.6s ease-out forwards;
    animation-delay: 2.6s;
}

.splash-content {
    text-align: center;
}

.splash-logo {
    width: 240px;
    max-width: 70vw;
    margin-bottom: 20px;
    animation: logoPop 1.4s ease-out forwards;
}

.splash-title {
    font-size: 2.3rem;
    font-weight: 700;
    color: #90ee90;
    opacity: 0;
    animation: fadeUp 1s ease-out forwards;
    animation-delay: 0.9s;
}

.splash-sub {
    font-size: 1.15rem;
    color: #cfcfcf;
    opacity: 0;
    animation: fadeUp 1s ease-out forwards;
    animation-delay: 1.3s;
}

/* 🔹 ADDITION (FIX) — USED BY LOGO BELOW */
.centered-img {
    display: block;
    margin-left: auto;
    margin-right: auto;
    border-radius: 12px;
}

@keyframes logoPop {
    0% { opacity: 0; transform: scale(0.5); }
    60% { opacity: 1; transform: scale(1.08); }
    100% { transform: scale(1); }
}

@keyframes fadeUp {
    0% { opacity: 0; transform: translateY(18px); }
    100% { opacity: 1; transform: translateY(0); }
}

@keyframes hideSplash {
    to { opacity: 0; visibility: hidden; }
}

/* ================= MAIN APP ================= */
.main {
    background-color: #0e1117;
}

.title {
    font-size: 2.5rem;
    font-weight: bold;
    text-align: center;
    color: #90ee90;
    margin-bottom: 0.5rem;
    animation: fadeInDown 1s ease-out;
}

.sub {
    font-size: 1.3rem;
    color: #cccccc;
    text-align: center;
    margin-bottom: 1.5rem;
    animation: fadeInUp 1s ease-in;
}

.card {
    background-color: #1c1c1c;
    padding: 1.2rem;
    border-radius: 12px;
    box-shadow: 0 4px 15px rgba(0,0,0,0.4);
    margin-bottom: 1rem;
}

.sidebar-note {
    font-size: 16px;
    font-weight: 500;
    background-color: #263238;
    padding: 12px;
    border-radius: 8px;
    color: #ffffff;
    border-left: 5px solid #4CAF50;
}

@keyframes fadeInDown {
    0% { opacity: 0; transform: translateY(-20px); }
    100% { opacity: 1; transform: translateY(0); }
}

@keyframes fadeInUp {
    0% { opacity: 0; transform: translateY(20px); }
    100% { opacity: 1; transform: translateY(0); }
}
</style>
""", unsafe_allow_html=True)

# ============================================================
# SPLASH SCREEN
# ============================================================
if logo_base64:
    st.markdown(f"""
    <div id="splash-screen">
        <div class="splash-content">
            <img src="data:image/png;base64,{logo_base64}" class="splash-logo">
            <div class="splash-title">Medicinal Plant Identifier</div>
            <div class="splash-sub">AI-Powered Plant Detection System</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ============================================================
# LOGO (EXISTING — NOW WORKING)
# ============================================================
logo_b64 = load_base64_image("assets/logo8.png")
if logo_b64:
    st.markdown(
        f'<img src="data:image/jpeg;base64,{logo_b64}" width="100" class="centered-img">',
        unsafe_allow_html=True
    )
else:
    st.warning("Logo image not found.")

# ============================================================
# MAIN CONTENT
# ============================================================
st.markdown('<div class="title">🌿 Medicinal Plant Identifier</div>', unsafe_allow_html=True)
st.markdown('<div class="sub">🔬 AI-powered Plant Identification & Enrichment System</div>', unsafe_allow_html=True)
st.markdown("<hr>", unsafe_allow_html=True)

agent = PredictionAgent()

# ============================================================
# 🚨 MANDATORY DISCLAIMER — KNOWN CLASSES ONLY
# ============================================================
st.markdown("""
<div class="card" style="border-left: 5px solid #ff8000;">
Welcome to the <b>Medicinal Plant Identifier</b>, a hybrid AI-powered tool that identifies
and explains medicinal plants using:
<ul>
    <li><b>Deep Learning (ResNet)</b></li>
    <li><b>QPSO-based Feature Optimization</b></li>
    <li><b>SVM Classification</b></li>
    <li><b>Wikipedia Enrichment</b></li>
</ul>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="card" style="border-left: 5px solid #ff9800;">
<b>⚠️ Known Classes Only Disclaimer</b><br><br>
This system is trained on <b>five selected medicinal plant species</b>:
<ul>
    <li>Neem</li>
    <li>Tulsi</li>
    <li>Aloe Vera</li>
    <li>Moringa</li>
    <li>Hibiscus</li>
</ul>
Predictions are <b>strictly restricted</b> to these classes only.
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="card" style="border-left: 5px solid #ff8500;">
<h4>🧠 Why this tool?</h4>
<ul>
    <li>Accurate leaf-based classification</li>
    <li>Scientific & medicinal insights</li>
    <li>Explainable AI predictions</li>
</ul>
</div>
""", unsafe_allow_html=True)

st.markdown(
    '<div class="sidebar-note">👈 Use the sidebar to upload a plant image and explore enriched AI predictions.</div>',
    unsafe_allow_html=True
)

# ============================================================
# CLEAN UI
# ============================================================
st.markdown("""
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

st.markdown("""<hr>
<center>© 2025 JDCOEM Medical Plant Identifier | Designed by Team LeafLogic</center>
""", unsafe_allow_html=True)
