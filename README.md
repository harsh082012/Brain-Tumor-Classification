# 🧠 Brain Tumor Classification using Deep Learning

## 🚀 Project Overview

This project focuses on **automatic classification of brain tumors from MRI images** using Deep Learning.
The model can classify images into **4 categories**:

* Glioma Tumor
* Meningioma Tumor
* Pituitary Tumor
* No Tumor

The system is built using **Transfer Learning (EfficientNetB3)** and deployed using **Streamlit**.

---

## 📂 Dataset Information

### 📌 Dataset Used

* Brain Tumor MRI Dataset (Kaggle)

### 📊 Dataset Details

* Total Images: **~3000+ images**
* Classes: **4**
* Image Types: MRI brain scans

### 🧩 Classes Distribution

* Glioma
* Meningioma
* Pituitary
* No Tumor

---

## ⚙️ Data Features & Preprocessing

* Images resized to **300x300**
* Normalization using EfficientNet preprocessing
* Data augmentation:

  * Rotation
  * Zoom
  * Horizontal flip
* Train / Validation / Test split

---

## 🧠 Model Architecture

* Base Model: **EfficientNetB3 (Pretrained on ImageNet)**
* Custom Classification Head:

  * GlobalAveragePooling
  * BatchNormalization
  * Dense Layers (1024 → 512 → 256)
  * Dropout (0.5, 0.4, 0.3)
* Output Layer:

  * Softmax (4 classes)

---

## 🔥 Training Strategy

### 🟢 Phase 1: Feature Extraction

* Base model **frozen**
* Only top layers trained
* Learning rate: `1e-4`

👉 Purpose:

* Learn dataset-specific patterns
* Prevent overfitting

---

### 🔵 Phase 2: Fine-Tuning

* Last **100 layers unfrozen**
* Lower learning rate: `5e-6`

👉 Purpose:

* Improve accuracy
* Fine-tune deep features

---

## 📈 Results

### ✅ Training Performance

* Validation Accuracy: **~94.88%**

### ✅ Test Performance

* Test Accuracy: **~91.75%**

---

## 📊 Classification Report (Summary)

* Overall Accuracy: **92%**
* Balanced precision and recall across classes

---

## 📉 Confusion Matrix Insights

* Strong performance on:

  * Pituitary
  * No Tumor
* Slight confusion between:

  * Glioma vs Meningioma

---

## ⚠️ Model Improvements

* Added **label smoothing**
* Used **dropout for regularization**
* Implemented **confidence threshold (0.85)**
* Added **invalid image detection** (rejects non-MRI images)

---

## 🖥️ Deployment

* Framework: **Streamlit**
* Model hosted on **Google Drive**
* Auto-download using `gdown`

---

## 🌐 Demo

👉 *(Add your Streamlit link here after deployment)*

---

## 📦 Installation

```bash
git clone https://github.com/your-username/brain-tumor-app.git
cd brain-tumor-app
pip install -r requirements.txt
streamlit run app.py
```

---

## 📌 Model Note

⚠️ Model file is not included in the repository due to size limitations.
It will be **automatically downloaded from Google Drive** when the app runs.

---

## 🧪 Features of the App

* Upload MRI image
* Predict tumor type
* Confidence score display
* Rejects non-medical images
* Fast and user-friendly UI

---

## 🛠️ Tech Stack

* Python
* TensorFlow / Keras
* OpenCV
* NumPy
* Streamlit

---

## 📚 Future Improvements

* Improve glioma vs meningioma classification
* Add Grad-CAM visualization
* Deploy using Docker
* Convert to mobile app

---

## 👨‍💻 Author

**Harsh**

---

## ⭐ If you like this project

Give it a ⭐ on GitHub!
