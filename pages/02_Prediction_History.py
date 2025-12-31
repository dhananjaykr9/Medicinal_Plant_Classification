import streamlit as st
import pandas as pd
import requests
from google.oauth2 import service_account
from google.auth.transport.requests import Request

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(page_title="Prediction History", layout="wide")
st.title("📁 Prediction History")
st.markdown(
    "Explore previously classified medicinal plants logged by the "
    "**ResNet50 + QPSO + SVM** prediction system."
)

# ============================================================
# FIREBASE REST CONFIG (USING secrets.toml [firebase])
# ============================================================
firebase_secrets = dict(st.secrets["firebase"])

PROJECT_ID = firebase_secrets["project_id"]
COLLECTION = "prediction_logs"

# Create service account credentials
credentials = service_account.Credentials.from_service_account_info(
    firebase_secrets,
    scopes=["https://www.googleapis.com/auth/datastore"]
)

credentials.refresh(Request())
ACCESS_TOKEN = credentials.token

FIRESTORE_URL = (
    f"https://firestore.googleapis.com/v1/projects/"
    f"{PROJECT_ID}/databases/(default)/documents/{COLLECTION}"
)

HEADERS = {
    "Authorization": f"Bearer {ACCESS_TOKEN}"
}

response = requests.get(FIRESTORE_URL, headers=HEADERS)
data = response.json()

# ============================================================
# PARSE FIRESTORE DOCUMENTS (MATCHES PREDICT PAGE)
# ============================================================
def parse_firestore_docs(docs):
    rows = []
    for doc in docs:
        fields = doc.get("fields", {})

        rows.append({
            "SVM Class": fields.get("SVM Class", {}).get("stringValue", "N/A"),
            "SVM Confidence": float(
                fields.get("SVM Confidence", {}).get("doubleValue", 0.0)
            ),
            "Scientific Name": fields.get("Scientific Name", {}).get("stringValue", "N/A"),
            "Genus": fields.get("Genus", {}).get("stringValue", "N/A"),
            "Family": fields.get("Family", {}).get("stringValue", "N/A"),
            "Wikipedia Summary": fields.get("Wikipedia Summary", {}).get("stringValue", ""),
            "Image URL": fields.get("Image URL", {}).get("stringValue", ""),
            "Timestamp": fields.get("Timestamp", {}).get("stringValue", ""),
        })

    return rows

docs = data.get("documents", [])
records = parse_firestore_docs(docs)

# ============================================================
# UI — FILTER & DISPLAY
# ============================================================
if records:
    df = pd.DataFrame(records)

    # ---------------- Sidebar Filters ----------------
    with st.sidebar:
        st.markdown("## 🔍 Filter Logs")

        plant_names = sorted(df["SVM Class"].unique())
        plant_filter = st.selectbox(
            "Plant Class",
            ["All Plants"] + plant_names
        )

        min_conf = st.slider(
            "Minimum Confidence",
            0.0, 1.0, 0.50, 0.01
        )

        sort_option = st.selectbox(
            "Sort By",
            [
                "Newest First",
                "Oldest First",
                "Confidence ↓",
                "Confidence ↑",
            ]
        )

        st.markdown("---")
        st.markdown("## 🔐 Admin Panel")
        admin_pass = st.text_input("Admin Password", type="password")
        is_admin = admin_pass == "admin@123"

    # ---------------- Filtering ----------------
    filtered_df = df[df["SVM Confidence"] >= min_conf]

    if plant_filter != "All Plants":
        filtered_df = filtered_df[filtered_df["SVM Class"] == plant_filter]

    if sort_option == "Newest First":
        filtered_df = filtered_df.sort_values("Timestamp", ascending=False)
    elif sort_option == "Oldest First":
        filtered_df = filtered_df.sort_values("Timestamp", ascending=True)
    elif sort_option == "Confidence ↓":
        filtered_df = filtered_df.sort_values("SVM Confidence", ascending=False)
    else:
        filtered_df = filtered_df.sort_values("SVM Confidence", ascending=True)

    # ---------------- Pagination ----------------
    PER_PAGE = 5
    total = len(filtered_df)
    pages = max(1, (total - 1) // PER_PAGE + 1)

    if pages > 1:
        page = st.slider("📄 Page", 1, pages, 1)
    else:
        page = 1  # only one page, no slider needed

    start = (page - 1) * PER_PAGE
    end = start + PER_PAGE

    st.markdown(
        f"### 🧬 Showing {start + 1}–{min(end, total)} of {total} records"
    )

    # ---------------- Display Cards ----------------
    for _, row in filtered_df.iloc[start:end].iterrows():
        col1, col2 = st.columns([1, 3])

        with col1:
            if row["Image URL"]:
                st.image(row["Image URL"], width=180)
            else:
                st.info("No image available")

        with col2:
            st.subheader(
                f"🧠 `{row['SVM Class']}` "
                f"({row['SVM Confidence'] * 100:.2f}%)"
            )
            st.markdown(
                f"""
- **Scientific Name:** `{row['Scientific Name']}`
- **Genus:** `{row['Genus']}`
- **Family:** `{row['Family']}`
- **Wikipedia:** {row['Wikipedia Summary'][:300]}...
- 🕒 **Timestamp:** `{row['Timestamp']}`
"""
            )

        st.divider()

    # ---------------- Export ----------------
    st.download_button(
        "⬇️ Download Filtered Logs (CSV)",
        filtered_df.to_csv(index=False),
        "prediction_logs.csv",
        "text/csv"
    )

    # ---------------- Admin Delete ----------------
    if is_admin:
        if st.button("🗑️ Delete ALL Logs from Firebase"):
            deleted = 0
            for doc in docs:
                doc_name = doc["name"]
                r = requests.delete(
                    f"https://firestore.googleapis.com/v1/{doc_name}",
                    headers=HEADERS
                )
                if r.status_code == 200:
                    deleted += 1
            st.success(f"✅ Deleted {deleted} logs successfully")
    else:
        st.info("Admin password required for deletion")

else:
    st.info("No prediction logs available yet.")

# ============================================================
# FOOTER
# ============================================================
st.markdown("<hr>", unsafe_allow_html=True)
st.caption("🛠 Team LeafLogic | Firebase-Backed Prediction History")
