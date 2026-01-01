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
# CSS STYLING + SPLASH SCREEN (ENHANCED ONLY)
# ============================================================
st.markdown("""
<style>

/* =============== GLOBAL THEME =============== */
html, body {
    background-color: #0e1117;
    color: #eaeaea;
    font-family: "Inter", sans-serif;
}

/* =============== SPLASH SCREEN =============== */
#about-splash {
    position: fixed;
    inset: 0;
    background: radial-gradient(circle at top, #0f2027, #0e1117 65%);
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
    font-size: 2.3rem;
    font-weight: 800;
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

/* =============== CONTENT STYLES =============== */
.centered-img {
    display: block;
    margin: 10px auto;
    border-radius: 14px;
}

.centered-gif {
    display: block;
    margin: 15px auto;
    width: 100%;
    max-height: 160px;
    object-fit: cover;
    border-radius: 18px;
}

/* Glass section */
.section {
    padding: 1.2rem 1.4rem;
    margin-bottom: 1.4rem;
    background: linear-gradient(145deg, #1a1d24, #14161c);
    border-left: 5px solid #4CAF50;
    border-radius: 10px;
    box-shadow: 0 6px 18px rgba(0,0,0,0.45);
}

/* Team Cards */
.team-card {
    border-radius: 14px;
    padding: 18px 16px;
    text-align: center;
    background: linear-gradient(145deg, #262730, #1f2028);
    color: #f1f1f1;
    margin: 8px;
    animation: fadeIn 1s ease-in-out;
    box-shadow: 0px 8px 20px rgba(0, 0, 0, 0.65);
    transition: transform 0.35s ease, box-shadow 0.35s ease;
}
.team-card:hover {
    transform: translateY(-8px) scale(1.02);
    box-shadow: 0px 14px 30px rgba(0,0,0,0.8);
}

.team-img {
    border-radius: 50%;
    border: 2px solid #4CAF50;
    margin-bottom: 10px;
    width: 100px;
    height: 100px;
    object-fit: cover;
}

a {
    color: #4FC3F7;
    text-decoration: none;
    margin: 0 2px;
}
a:hover {
    text-decoration: underline;
}

@keyframes fadeIn {
    0% { opacity: 0; transform: translateY(25px); }
    100% { opacity: 1; transform: translateY(0); }
}

</style>
""", unsafe_allow_html=True)

# ============================================================
# SPLASH SCREEN HTML
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
# LOGO
# ============================================================
logo_b64 = load_base64_image("assets/logo8.png")
if logo_b64:
    st.markdown(
        f'<img src="data:image/jpeg;base64,{logo_b64}" width="110" class="centered-img">',
        unsafe_allow_html=True
    )

# ============================================================
# BANNER GIF
# ============================================================
gif_b64 = load_base64_image("assets/plant7.gif")
if gif_b64:
    st.markdown(
        f'<img src="data:image/gif;base64,{gif_b64}" class="centered-gif">',
        unsafe_allow_html=True
    )

# ============================================================
# TITLE & INTRO
# ============================================================
st.title("📑 About the Project")

st.markdown("""
<div class="section">
<h3>🌿 Advanced Medicinal Plant Identification System</h3>
<p>
This project is a hybrid AI-powered solution designed to <b>identify medicinal plants</b>
using image classification, enriched with biological intelligence and public knowledge APIs.
</p>
</div>
""", unsafe_allow_html=True)

# ============================================================
# TEAM MEMBERS (UNCHANGED LOGIC)
# ============================================================
st.markdown("### 👨🏻‍💻 Team Members")

teams = [
    {
        "name": "Dr. S. V. Sonekar",
        "role": "Project Mentor & Guide",
        "class": "Principal",
        "destination": "JDCOEM, Nagpur",
        "photo": "assets/team/principal_sir1.png",
        "linkedin": "https://www.linkedin.com/in/shrikant-sonekar-52b393ba?originalSubdomain=in",
        "email": "mailto:principal@jdcoem.ac.in",
        "img_size": 95
    },
    {
        "name": "Dipanshu Likhar",
        "role": "Data Handling & Processing",
        "class": "Student",
        "destination": "JDCOEM, Nagpur",
        "photo": "assets/team/Dipanshu.png",
        "linkedin": "https://www.linkedin.com/in/dipanshulikhar/?originalSubdomain=in",
        "email": "mailto:likhardipanshu@gmail.com",
        "img_size": 95
    },
    {
        "name": "Dhananjay Kharkar",
        "role": "Developer, Pipeline Handling",
        "class": "Student",
        "destination": "JDCOEM, Nagpur",
        "photo": "assets/team/Dhananjay.png",
        "linkedin": "https://www.linkedin.com/in/dhananjaykharkar/",
        "email": "mailto:dkharkar00@gmail.com",
        "img_size": 95
    },
    {
        "name": "Prajwal Khapekar",
        "role": "Backend + Integration & Database",
        "class": "Student",
        "destination": "JDCOEM, Nagpur",
        "photo": "assets/team/prajwal.png",
        "linkedin": "https://www.linkedin.com/in/prajwal-khapekar-71618a277/",
        "email": "prajwalkhapekar96@gmail.com",
        "img_size": 95
    },
    {
        "name": "Harshal Chhatri",
        "role": "Technical & Documentation",
        "class": "Student",
        "destination": "JDCOEM, Nagpur",
        "photo": "assets/team/harshal1.png",
        "linkedin": "https://www.linkedin.com/in/harshal-chhatri-767089224/",
        "email": "mailto:harshalchhatri231@gmail.com",
        "img_size": 95
    }
]

cols = st.columns(len(teams))
for i, member in enumerate(teams):
    photo_b64 = load_base64_image(member["photo"])
    img_size = member.get("img_size", 100)

    with cols[i]:
        st.markdown(f"""
        <div class="team-card">
            <img src="data:image/jpeg;base64,{photo_b64}"
                 class="team-img"
                 style="width:{img_size}px; height:{img_size}px;">
            <b>{member['name']}</b><br>
            {member['class']}<br>
            <small>{member['destination']}</small><br>
            <i>{member['role']}</i><br>
            <a href="{member['linkedin']}" target="_blank">ℹ️</a>
            <a href="{member['email']}" target="_blank">✉️</a>
        </div>
        """, unsafe_allow_html=True)

# ============================================================
# REMAINING SECTIONS (UNCHANGED CONTENT)
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
| Wikipedia | Knowledge enrichment |
| Streamlit | Web deployment |

---

### 🧪 How It Works
1. Upload leaf image  
2. CNN extracts features  
3. QPSO optimizes features  
4. SVM predicts plant  
5. Enrich knowledge  
6. Dashboard visualizes results  

---

### 📢 Acknowledgments
We thank **JDCOEM, Nagpur** for support and mentorship.

""")

# ============================================================
# FOOTER
# ============================================================
st.markdown("<hr>", unsafe_allow_html=True)
st.caption("© Team LeafLogic | AI System for Medicinal Plant Identification")

