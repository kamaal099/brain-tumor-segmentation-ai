# 🧠 Brain AI — Brain Tumor Segmentation System

An end-to-end AI system that automatically detects and segments brain tumors from MRI images using deep learning.

---

# 📌 Overview

Brain tumors are critical medical conditions that require precise diagnosis. Manual analysis of MRI scans is time-consuming and depends heavily on expert interpretation.

👉 This project builds an **AI-powered solution** that:

* Detects tumor regions automatically
* Generates segmentation masks
* Visualizes tumor areas with overlays

---

# 🎯 Objective

* Automate brain tumor detection
* Improve diagnostic efficiency
* Provide consistent segmentation results
* Demonstrate real-world AI application in healthcare

---

# 🧠 How It Works (Simple Explanation)

1. User uploads an MRI image
2. Image is preprocessed (resized, normalized)
3. Deep learning model analyzes the image
4. Tumor region is predicted pixel-by-pixel
5. Output is displayed as:

   * Predicted mask
   * Overlay on original MRI
   * Confidence score

---

# ⚙️ Technology Stack

* **Backend:** FastAPI
* **Model:** TensorFlow / Keras
* **Frontend:** HTML, CSS, JavaScript
* **Image Processing:** OpenCV, NumPy

---

# 🧠 Model Details

## Architecture: U-Net

U-Net is a convolutional neural network designed specifically for **image segmentation tasks**.

### Why U-Net?

* Provides pixel-level predictions
* Works well with medical images
* Retains spatial information using skip connections
* Performs better than classification models (ResNet, VGG) for segmentation

---

# 🧠 Type of Learning

This is a **Supervised Learning** problem.

* Input: MRI images
* Output: Ground truth tumor masks
* The model learns mapping between them

---

# 📊 Dataset

* **BraTS 2020 (Brain Tumor Segmentation Challenge)**
* Multi-modal MRI scans:

  * T1
  * T1ce
  * T2
  * FLAIR
* Includes labeled tumor segmentation masks

---

# 🔄 Pipeline

1. Data loading
2. Preprocessing
3. Model training
4. Prediction
5. Post-processing
6. Visualization

---

# 📊 Performance

* Dice Score: **~0.48**

---

# ⚠️ Why Accuracy is Moderate (~0.48)

* Uses 2D slices instead of full 3D MRI volumes
* Tumor region is very small (class imbalance)
* Basic loss function used
* Limited training optimization
* No advanced enhancements (attention, transfer learning)

---

# ⚖️ Pros & Cons

## ✅ Advantages

* Real-world medical use case
* Uses standard dataset (BraTS)
* End-to-end pipeline (data → model → UI)
* Deployable backend system

## ❌ Limitations

* Moderate accuracy
* Not suitable for clinical use
* Uses single image instead of multi-modal input in UI
* Lacks advanced model optimization

---

# 🖥 Project Structure

```bash
brain-tumor-segmentation-ai/
│
├── backend/
│   ├── app.py
│   ├── utils.py
│   └── requirements.txt
│
├── frontend/
│   ├── index.html
│   ├── script.js
│   ├── style.css
│   └── assets/
│
├── README.md
├── .gitignore
└── LICENSE
```

---

# 🚀 How to Run

## 1. Backend

```bash
cd backend
pip install -r requirements.txt
uvicorn app:app --reload
```

---

## 2. Frontend

Open `frontend/index.html` in your browser.

---

# 📸 Output Example

The system produces:

* Input MRI image
* Predicted tumor mask
* Overlay visualization

(Add your screenshot here)

---

# 📌 Important Note

⚠️ This project is a **research prototype** and should not be used for real medical diagnosis.

---

# 🔮 Future Improvements

* Implement 3D U-Net
* Use Dice Loss / Focal Loss
* Add attention mechanisms
* Improve dataset preprocessing
* Integrate explainable AI (XAI)
* Deploy with real hospital data

---

# 🎓 Academic Value

This project demonstrates:

* Deep learning (CNN, U-Net)
* Medical image processing
* Model deployment (API + frontend)
* End-to-end system design

---

# 🏆 Conclusion

This project showcases how AI can assist in medical imaging by automating tumor segmentation. While improvements are needed for higher accuracy, it provides a strong foundation for real-world healthcare applications.

---

# 🙌 Author

Final Year Machine Learning Project
Brain Tumor Segmentation using Deep Learning
