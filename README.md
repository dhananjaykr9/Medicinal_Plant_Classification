# 🌿 LeafLogic – Medicinal Plant Identification System

LeafLogic is an **AI-powered medicinal plant identification system** that classifies leaf images using a hybrid deep learning and optimization framework. The system combines **ResNet50-based deep feature extraction**, **Quantum-behaved Particle Swarm Optimization (QPSO)** for feature selection, and **Support Vector Machine (SVM)** classification.

To enhance transparency and trust, the system integrates **Explainable AI (XAI)** techniques such as **Grad-CAM** (spatial explanation) and **SHAP** (feature-level explanation). A **Streamlit-based dashboard** enables real-time interaction, while **Firebase** is used for prediction logging and user feedback collection.

---

## 🚀 Features

* 🌱 **Medicinal Plant Classification**

  * Identifies 5 medicinal plant species:

    * Neem
    * Tulsi
    * Aloe Vera
    * Moringa
    * Hibiscus

* 🧠 **Hybrid AI Pipeline**

  * ResNet50 (fine-tuned) for deep feature extraction
  * QPSO for optimal feature selection
  * SVM (RBF kernel) for robust classification

* 🔍 **Explainable AI**

  * **Grad-CAM**: Highlights image regions influencing predictions
  * **SHAP**: Feature-level contribution analysis (optimized for speed)

* 🧑‍⚕️ **Human-in-the-Loop Feedback**

  * Users can mark predictions as **Correct / Incorrect**
  * Feedback stored for future dataset expansion

* 📊 **Prediction History Dashboard**

  * View previous predictions
  * Filter by class, confidence, timestamp
  * Displays user feedback (Yes / No)

* ☁️ **Cloud Ready**

  * Deployed on **Streamlit Cloud**
  * Firebase Firestore for secure logging

---

## 🧪 System Architecture

```
Leaf Image
   ↓
ResNet50 (Fine-Tuned CNN)
   ↓
Global Average Pooling
   ↓
QPSO Feature Selection
   ↓
SVM (RBF Kernel)
   ↓
Prediction + Confidence
   ↓
Grad-CAM & SHAP Explainability
   ↓
Streamlit Dashboard + Firebase Logging
```

---

## 🖥️ Technologies Used

### Machine Learning & AI

* TensorFlow / Keras
* Scikit-learn
* SHAP
* OpenCV
* NumPy

### Backend & Database

* Firebase Firestore
* MongoDB (Plant metadata)

### Frontend & Deployment

* Streamlit
* Streamlit Cloud

---

## 📦 Software Requirements

* Python **3.10**
* Required Python Libraries:

  ```txt
  streamlit
  tensorflow==2.13.0
  numpy
  opencv-python-headless
  pillow
  scikit-learn
  joblib
  pymongo
  firebase-admin
  certifi
  shap
  ```

> ⚠️ **Note:** TensorFlow 2.13 is compatible with Python 3.10.
> Do **not** use Python 3.12+ for this project.

---

## 📁 Project Structure

```
medicinal_plant_classification/
│
├── Home.py
├── pages/
│   ├── 01_Upload & Predict.py
│   └── 02_Prediction History.py
│
├── models/
│   └── resnet_finetuned_model_tf.keras
│
├── models_finetuned/
│   ├── qpso_svm_model_finetuned.pkl
│   └── qpso_scaler_finetuned.pkl
│
├── qpso_finetuned/
│   └── qpso_selected_mask_finetuned.npy
│
├── assets
│
├── requirements.txt
├── runtime.txt
└── README.md
```

---

## ⚙️ Installation and Setup

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/yourusername/medicinal_plant_classification.git
cd medicinal_plant_classification
```

### 2️⃣ Create Virtual Environment

```bash
python -m venv plant_env
source plant_env/bin/activate   # Linux / Mac
plant_env\Scripts\activate      # Windows
```

### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 4️⃣ Configure Firebase

* Create a Firebase project
* Enable **Firestore**
* Add Firebase credentials to:

  ```toml
  .streamlit/secrets.toml
  ```

### 5️⃣ Run the App

```bash
streamlit run Home.py
```

---

## 📊 Explainability Module

### 🔥 Grad-CAM (Spatial Explanation)

* Visualizes which regions of the leaf image influenced the prediction
* Helps validate model focus on leaf venation and morphology

### 🔎 SHAP (Feature-Level Explanation)

* Explains how selected deep features influence SVM predictions
* Optimized for performance using:

  * Reduced feature subset
  * Cached background samples

---

## 🧪 Experimental Setup

* Dataset: Medicinal plant leaf images (5 classes)
* Image Size: 224 × 224
* Feature Vector: ResNet50 GAP features
* Feature Selection: QPSO
* Classifier: SVM with RBF kernel
* Evaluation Metrics:

  * Accuracy
  * Precision
  * Recall
  * F1-score
  * ROC–AUC
  * Top-3 Accuracy

---

## 📈 Results (Summary)

* Fine-tuned ResNet50 improves domain adaptation
* QPSO reduces feature redundancy and dimensionality
* SVM classifier provides stable generalization
* Explainability modules enhance trust and transparency

---

## 🧩 Limitations

* Limited to 5 medicinal plant species
* Performance depends on image quality
* SHAP explanations are approximate due to optimization
* Not intended for medical diagnosis

---

## 🤝 Contributions

Contributions are welcome!
You can:

* Improve model performance
* Add new plant species
* Enhance UI/UX
* Optimize SHAP further

Please fork the repository and submit a pull request.

---

## 🙏 Acknowledgments

* TensorFlow and Scikit-learn communities
* SHAP and Grad-CAM research contributors
* Open-source plant datasets
* Academic guidance and institutional support

---

## 📬 Contact

For questions or suggestions, feel free to open an issue or contact:

**Dhananjay Kharkar**
GitHub: [https://github.com/dhananjaykr9](https://github.com/dhananjaykr9)

---
