import streamlit as st
import pandas as pd
import requests
from google.oauth2 import service_account
from google.auth.transport.requests import Request
import firebase_admin
from firebase_admin import credentials, firestore
from datetime import datetime


# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(page_title="Prediction History", layout="wide")

# ============================================================
# GLOBAL UI STYLES
# ============================================================
st.markdown(
    """
<style>
body { background-color: #0e1117; }
.history-card {
    background-color: #121212;
    border-radius: 16px;
    padding: 18px;
    border: 1px solid #2a2a2a;
    box-shadow: 0 6px 14px rgba(0,0,0,0.45);
    margin-bottom: 18px;
}
.history-title {
    color: #7CFF9B;
    font-size: 1.25rem;
    font-weight: 600;
    margin-bottom: 8px;
}
.history-row {
    font-size: 0.95rem;
    color: #dddddd;
    margin-bottom: 6px;
}
.feedback-yes { color: #7CFF9B; font-weight: 700; }
.feedback-no { color: #ff6b6b; font-weight: 700; }
.feedback-na { color: #aaaaaa; font-weight: 700; }
.analytics-card {
    background-color: #101418;
    border-radius: 14px;
    padding: 16px;
    border: 1px solid #2a2a2a;
    text-align: center;
}
.analytics-value {
    font-size: 1.6rem;
    font-weight: 700;
    color: #7CFF9B;
}
.analytics-label {
    font-size: 0.85rem;
    color: #bbbbbb;
}
</style>
""",
    unsafe_allow_html=True
)

# ============================================================
# HEADER
# ============================================================
st.title("📁 Prediction History")
st.markdown(
    "Explore previously classified medicinal plants logged by the "
    "**ResNet50 + QPSO + SVM** production system."
)
st.divider()

# ============================================================
# 🔐 ROLE SELECTION
# ============================================================
with st.sidebar:
    st.markdown("## 👤 Access Role")
    role = st.radio(
        "Select role",
        ["viewer", "analyst", "admin"],
        help="Viewer: read-only | Analyst: analytics & export | Admin: full access"
    )

    role_password = None
    is_analyst = False
    is_admin = False

    if role in ["analyst", "admin"]:
        role_password = st.text_input("Enter role password", type="password")

        if role == "analyst" and role_password == st.secrets["roles"]["analyst_password"]:
            is_analyst = True

        if role == "admin" and role_password == st.secrets["roles"]["admin_password"]:
            is_admin = True
            is_analyst = True  # admin inherits analyst permissions

    if role == "viewer":
        st.info("Viewer mode: read-only access")

# ============================================================
# FIREBASE REST CONFIG
# ============================================================
firebase_secrets = dict(st.secrets["firebase"])
PROJECT_ID = firebase_secrets["project_id"]
PRED_COLLECTION = "prediction_logs"

credentials_rest = service_account.Credentials.from_service_account_info(
    firebase_secrets,
    scopes=["https://www.googleapis.com/auth/datastore"]
)
credentials_rest.refresh(Request())
HEADERS = {"Authorization": f"Bearer {credentials_rest.token}"}

@st.cache_resource
def init_admin_firestore():
    if not firebase_admin._apps:
        cred = credentials.Certificate(dict(st.secrets["firebase"]))
        firebase_admin.initialize_app(cred)
    return firestore.client()

admin_db = init_admin_firestore()

# ============================================================
# FETCH PREDICTION LOGS
# ============================================================
PRED_URL = (
    f"https://firestore.googleapis.com/v1/projects/"
    f"{PROJECT_ID}/databases/(default)/documents/{PRED_COLLECTION}"
)
pred_data = requests.get(PRED_URL, headers=HEADERS).json()

# ============================================================
# PARSE DOCUMENTS
# ============================================================
def parse_firestore_docs(docs):
    rows = []
    for doc in docs:
        fields = doc.get("fields", {})

        ts_str = fields.get("Timestamp", {}).get("stringValue", "")

        # ✅ SAFE datetime parsing (CRITICAL FIX)
        try:
            ts_dt = datetime.fromisoformat(ts_str.replace("Z", ""))
        except Exception:
            ts_dt = datetime.min

        rows.append({
            "SVM Class": fields.get("SVM Class", {}).get("stringValue", "N/A"),
            "SVM Confidence": float(fields.get("SVM Confidence", {}).get("doubleValue", 0.0)),
            "Scientific Name": fields.get("Scientific Name", {}).get("stringValue", "N/A"),
            "Genus": fields.get("Genus", {}).get("stringValue", "N/A"),
            "Family": fields.get("Family", {}).get("stringValue", "N/A"),
            "Wikipedia Summary": fields.get("Wikipedia Summary", {}).get("stringValue", ""),
            "Image URL": fields.get("Image URL", {}).get("stringValue", ""),
            "Timestamp": ts_str,        # ✅ keep original (UI)
            "Timestamp_dt": ts_dt,      # ✅ new (sorting)
            "User Feedback": fields.get("User Feedback", {}).get("stringValue", "Not Provided"),
        })
    return rows

records = parse_firestore_docs(pred_data.get("documents", []))

# ============================================================
# ANALYTICS (ANALYST + ADMIN)
# ============================================================
if records:
    df = pd.DataFrame(records)

    if is_analyst:
        total = len(df)
        yes_cnt = (df["User Feedback"].str.lower() == "yes").sum()
        no_cnt = (df["User Feedback"].str.lower() == "no").sum()
        na_cnt = total - yes_cnt - no_cnt

        st.markdown("## 📊 Feedback Analytics")

        c1, c2, c3, c4 = st.columns(4)
        c1.markdown(f"<div class='analytics-card'><div class='analytics-value'>{total}</div><div class='analytics-label'>Total</div></div>", unsafe_allow_html=True)
        c2.markdown(f"<div class='analytics-card'><div class='analytics-value'>{yes_cnt}</div><div class='analytics-label'>Correct</div></div>", unsafe_allow_html=True)
        c3.markdown(f"<div class='analytics-card'><div class='analytics-value'>{no_cnt}</div><div class='analytics-label'>Incorrect</div></div>", unsafe_allow_html=True)
        c4.markdown(f"<div class='analytics-card'><div class='analytics-value'>{na_cnt}</div><div class='analytics-label'>Not Provided</div></div>", unsafe_allow_html=True)

        st.divider()

# ============================================================
# FILTERING (NO CONFIDENCE FILTER)
# ============================================================
with st.sidebar:
    st.markdown("## 🔍 Filter Logs")
    plant_filter = st.selectbox(
        "Plant Class",
        ["All Plants"] + sorted(df["SVM Class"].unique())
    )
    sort_option = st.selectbox(
        "Sort By",
        ["Newest First", "Oldest First", "Confidence ↓", "Confidence ↑"]
    )

filtered_df = df.copy()

if plant_filter != "All Plants":
    filtered_df = filtered_df[filtered_df["SVM Class"] == plant_filter]

if sort_option == "Newest First":
    filtered_df = filtered_df.sort_values("Timestamp_dt", ascending=False)
elif sort_option == "Oldest First":
    filtered_df = filtered_df.sort_values("Timestamp_dt", ascending=True)
elif sort_option == "Confidence ↓":
    filtered_df = filtered_df.sort_values("SVM Confidence", ascending=False)
else:
    filtered_df = filtered_df.sort_values("SVM Confidence", ascending=True)

# ============================================================
# DISPLAY CARDS
# ============================================================
for _, row in filtered_df.iterrows():
    fb = row["User Feedback"].lower()
    fb_class = "feedback-yes" if fb == "yes" else "feedback-no" if fb == "no" else "feedback-na"

    st.markdown(
        f"""
<div class="history-card">
    <div class="history-title">🧠 {row['SVM Class']} ({row['SVM Confidence']*100:.2f}%)</div>
    <div style="display:flex; gap:16px;">
        <div>{"<img src='"+row["Image URL"]+"' width='160' style='border-radius:12px'/>" if row["Image URL"] else "No image"}</div>
        <div>
            <div class="history-row"><b>Scientific:</b> {row['Scientific Name']}</div>
            <div class="history-row"><b>Genus:</b> {row['Genus']}</div>
            <div class="history-row"><b>Family:</b> {row['Family']}</div>
            <div class="history-row"><b>User Feedback:</b>
                <span class="{fb_class}">{row['User Feedback'].upper()}</span>
            </div>
        </div>
    </div>
</div>
""",
        unsafe_allow_html=True
    )

# ============================================================
# EXPORT (ANALYST + ADMIN)
# ============================================================
if is_analyst:
    st.download_button(
        "⬇️ Download Logs (CSV)",
        filtered_df.to_csv(index=False),
        "prediction_logs.csv",
        "text/csv"
    )

# ============================================================
# DELETE (ADMIN ONLY)
# ============================================================
if is_admin:
    st.markdown("### ⚠️ Admin Controls")
    if st.button("🗑️ Delete ALL Logs from Firebase"):
        for doc in admin_db.collection(PRED_COLLECTION).stream():
            doc.reference.delete()
        st.success("✅ All prediction logs deleted")

# ============================================================
# FOOTER
# ============================================================
st.divider()
st.caption("© Team LeafLogic | AI System for Medicinal Plant Identification")
