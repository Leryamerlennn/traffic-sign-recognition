# TrafficSignsAI — Traffic Sign Detection & Recognition Pipeline

This project implements a complete end-to-end pipeline for detecting and recognizing traffic signs using a hybrid approach: classical computer vision for fast sign localization and a lightweight CNN classifier for identifying 43 traffic sign categories. The system is optimized to run efficiently on a standard CPU while remaining robust to real-world conditions such as night scenes, glare, rain, and noise.

---

## 🧠 Project Goal

To develop a full pipeline that:
1. **Locates traffic signs** using contour-based classical CV.
2. **Crops, resizes, and normalizes ROIs** to a consistent 32×32 tensor format.
3. **Classifies** each sign into one of **43 categories** using a compact CNN.
4. Ensures **real-time performance** and **high robustness** on everyday road images.

---

## 🔎 Pipeline Overview

### **1. Input**
- Real-life frame from a webcam or video stream.

### **2. CV Detector**
- Extract contours from the input frame.  
- Compute bounding boxes + expand edges slightly.  
- Crop ROIs from the original image.  
- Resize each ROI to **32×32**.  
- Normalize and convert to tensor input.

### **3. CNN Classifier**
- Lightweight VGG-style CNN with SE-block attention.  
- Two Conv–Conv–Pool feature extractor blocks.  
- Dense layer + Dropout for regularization.  
- Softmax output over **44 traffic sign classes**.

### **4. Output**
- Classification accuracy: **97–98%**  
- Detection quality (classical CV): **mAP@0.5 ≈ 0.82**  
- End-to-end pipeline accuracy: **≈ 90%**  
- Robustness to real-world conditions: **80–85%**

---

## 📚 Dataset Engineering

We created a unified dataset by merging **three independent sources** into a single 160k+ image dataset with consistent labels across 43 classes.

Processing steps included:
- Noise cleaning & duplicate removal  
- Resize normalization (32×32 and 96×96)  
- Strong augmentations: rotations, blur, brightness/contrast shifts, hue changes, color jitter  
- Label unification and balanced splitting

This significantly improved robustness to difficult scenes.

---

## 🎯 Key Insights

- Detection is the main bottleneck: inaccurate ROI → unavoidable CNN errors.  
- SE-blocks improved accuracy without increasing model complexity.  
- Augmentations were essential for robustness to night, glare, rain, and noise.  
- The full pipeline performs reliably on real road images.

---

## 👥 Team

- **Valeria Neganova** — CNN classifier development  [v.neganova@innopolis.university](mailto:v.neganova@innopolis.university)
- **Anastasia Malakhova** — CV detector development  [a.malakhova@innopolis.university](mailto:a.malakhova@innopolis.university)
- **Nikolay Rostov** — Dataset engineering, integration, testing  [n.rostov@innopolis.university](mailto:n.rostov@innopolis.university)

---

## 🛠️ Installation

```bash
pip install -r requirements.txt
```

## ▶️ Usage

```bash
python run_pipeline.py
```



