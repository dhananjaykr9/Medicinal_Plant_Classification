import streamlit as st
from PIL import Image
import base64
import os

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="About | Medicinal Plant AI",
    layout="centered"
)

# ============================================================
# HELPER: SAFE BASE64 LOADER
# ============================================================
def load_base64_image(path):
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()

# ============================================================
# LOAD SPLASH LOGO (REUSED SAFELY)
# ============================================================
splash_logo_b64 = load_base64_image("assets/logo_leaflogic.png")

# ============================================================
# CSS STYLING + SPLASH SCREEN
# ============================================================
st.markdown("""
<style>

/* ================= SPLASH SCREEN ================= */
#about-splash {
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

.splash-inner {
    text-align: center;
}

.splash-logo {
    width: 220px;
    max-width: 75vw;
    margin-bottom: 18px;
    animation: logoPop 1.4s ease-out forwards;
}

.splash-title {
    font-size: 2.2rem;
    font-weight: 700;
    color: #90ee90;
    opacity: 0;
    animation: fadeUp 1s ease-out forwards;
    animation-delay: 0.9s;
}

.splash-sub {
    font-size: 1.1rem;
    color: #cfcfcf;
    opacity: 0;
    animation: fadeUp 1s ease-out forwards;
    animation-delay: 1.3s;
}

/* Animations */
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

/* ================= EXISTING UI ================= */
.centered-img {
    display: block;
    margin-left: auto;
    margin-right: auto;
    border-radius: 12px;
}

.team-card {
    border-radius: 12px;
    padding: 15px;
    text-align: center;
    background-color: #262730;
    color: #f1f1f1;
    margin: 10px;
    animation: fadeIn 1s ease-in-out;
    box-shadow: 0px 4px 12px rgba(0, 0, 0, 0.6);
    transition: transform 0.3s ease;
}
.team-card:hover {
    transform: translateY(-5px);
}

.team-img {
    border-radius: 50%;
    border: 2px solid #4CAF50;
    margin-bottom: 10px;
    width: 100px;
    height: 100px;
    object-fit: cover;
}

.section {
    padding: 1rem;
    margin-bottom: 1rem;
    background-color: #1e1e1e;
    border-left: 5px solid #4CAF50;
    border-radius: 6px;
}

a {
    color: #4FC3F7;
    text-decoration: none;
}
a:hover {
    text-decoration: underline;
}

@keyframes fadeIn {
    0% { opacity: 0; transform: translateY(20px); }
    100% { opacity: 1; transform: translateY(0); }
}

.centered-gif {
    display: block;
    margin-left: auto;
    margin-right: auto;
    width: 100%;
    max-height: 150px;
    object-fit: cover;
    border-radius: 16px;
}
</style>
""", unsafe_allow_html=True)

# ============================================================
# SPLASH SCREEN HTML (BASE64 — GUARANTEED)
# ============================================================
if splash_logo_b64:
    st.markdown(f"""
    <div id="about-splash">
        <div class="splash-inner">
            <img src="data:image/png;base64,{splash_logo_b64}" class="splash-logo">
            <div class="splash-title">About LeafLogic</div>
            <div class="splash-sub">Medicinal Plant AI System</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ============================================================
# LOGO (EXISTING — UNCHANGED)
# ============================================================
logo_b64 = load_base64_image("assets/logo.png")
if logo_b64:
    st.markdown(
        f'<img src="data:image/jpeg;base64,{logo_b64}" width="250" class="centered-img">',
        unsafe_allow_html=True
    )
else:
    st.warning("Logo image not found.")

# ============================================================
# BANNER GIF (EXISTING — UNCHANGED)
# ============================================================
gif_b64 = load_base64_image("assets/leaflogic.gif")
if gif_b64:
    st.markdown(
        f'<img src="data:image/gif;base64,{gif_b64}" class="centered-gif">',
        unsafe_allow_html=True
    )
else:
    st.warning("Banner GIF not found.")

# ============================================================
# TITLE & INTRO
# ============================================================
st.title("📑 About the Project")

st.markdown("""
<div class="section">
<h3>🌿 Advanced Medicinal Plant Detection System</h3>
<p>
This project is a hybrid AI-powered solution designed to <b>identify medicinal plants</b>
using image classification, with advanced enrichment from biological and public data sources.
</p>
</div>
""", unsafe_allow_html=True)

# ============================================================
# TEAM MEMBERS (UNCHANGED)
# ============================================================
st.markdown("### 👨🏻‍💻 Team Members")

teams = [
    {
        "name": "Dhananjay Kharkar",
        "role": "Lead Developer, IoT + AI",
        "class": "Student",
        "destination": "JDCOEM, Nagpur",
        "photo": "assets/team/dhananjay1.jpg",
        "linkedin": "https://www.linkedin.com/in/dhananjaykharkar/",
        "email": "mailto:dkharkar00@gmail.com"
    },
    {
        "name": "Dipanshu Likhar",
        "role": "Backend + Integration & Database",
        "class": "Student",
        "destination": "JDCOEM, Nagpur",
        "photo": "assets/team/dipanshu.jpg",
        "linkedin": "#",
        "email": "mailto:aditi@vnit.ac.in"
    },
    {
        "name": "Prjawal Khapekar",
        "role": "Data Handling & Processing",
        "class": "Student",
        "destination": "JDCOEM, Nagpur",
        "photo": "assets/team/prajwal.jpg",
        "linkedin": "https://www.linkedin.com/in/prajwal-khapekar-71618a277/",
        "email": "prajwalkhapekar96@gmail.com"
    },
    {
        "name": "Harshal Chhatri",
        "role": "Technical & Documentation",
        "class": "Student",
        "destination": "JDCOEM, Nagpur",
        "photo": "assets/team/harshal.jpg",
        "linkedin": "https://www.linkedin.com/in/harshal-chhatri-767089224/",
        "email": "mailto:harshalchhatri231@gmail.com"
    },
    {
        "name": "Dr. S. V. Sonekar",
        "role": "Faculty Mentor & Guide",
        "class": "Principal",
        "destination": "JDCOEM, Nagpur",
        "photo": "assets/team/principal_sir.jpg",
        "linkedin": "#",
        "email": "mailto:principal@jdcoem.ac.in"
    }
]

cols = st.columns(len(teams))
for i, member in enumerate(teams):
    photo_b64 = load_base64_image(member["photo"])
    with cols[i]:
        st.markdown(f"""
        <div class="team-card">
            <img src="data:image/jpeg;base64,{photo_b64}" class="team-img">
            <b>{member['name']}</b><br>
            {member['class']}<br>
            <small>{member['destination']}</small><br>
            <i>{member['role']}</i><br>
            <a href="{member['linkedin']}" target="_blank">ℹ️</a>
            <a href="{member['email']}" target="_blank">✉️</a>
        </div>
        """, unsafe_allow_html=True)

# ============================================================
# REMAINING SECTIONS
# ============================================================
st.markdown("""
---

### 🎯 Objective
To provide an accurate, explainable, and educational plant identification system for:
- Botanists
- Ayurvedic researchers
- Farmers & forest officials
- Educational institutions

---

### 🧠 Technology Stack
| Component | Description |
|---------|-------------|
| ResNet-50 | CNN feature extraction |
| QPSO | Feature selection |
| SVM | Classification |
| Wikipedia API | Knowledge enrichment |
| Streamlit | Web deployment |

---

### 🧪 How It Works
1. Upload leaf image  
2. CNN extracts features  
3. QPSO optimizes features  
4. SVM predicts plant  
5. APIs enrich knowledge  
6. Dashboard visualizes results  

---

### 📢 Acknowledgments
We thank **JDCOEM Nagpur** for support and mentorship.

---

📬 Visit our GitHub or contact the team for collaboration.
""")

# ============================================================
# FOOTER
# ============================================================
st.markdown("<hr>", unsafe_allow_html=True)
st.caption("🛠 Developed by Team LeafLogic | 🚀 Patent-ready Project")
