# Traffic Sign Detection and Recognition

This project develops a complete pipeline for detecting and recognizing traffic signs in images using deep learning and computer vision techniques.



## 🧠 Project Goal

To build a system that can:
1. **Detect traffic signs** in road images using object detection models.
2. **Classify** each detected sign into its correct category (e.g., speed limit, stop sign, pedestrian crossing).
3. Work reliably under **various conditions** (lighting, occlusion, rotation).



## 🎯 Why It Matters

Traffic sign recognition is crucial for:
- **Autonomous vehicles** and driver-assistance systems (ADAS)
- **Road safety** applications
- **Navigation** and smart infrastructure
- **Mobile apps** (e.g., for driving learners or assistance for visually impaired users)



## 👥 Team

- **Neganova Valeria** – [v.neganova@innopolis.university](mailto:v.neganova@innopolis.university)
- **Malakhova Anastasia** – [a.malakhova@innopolis.university](mailto:a.malakhova@innopolis.university)
- **Nikolay Rostov** – [n.rostov@innopolis.university](mailto:n.rostov@innopolis.university)



## 🔗 Datasets

We are working with:
- **GTSDB** (German Traffic Sign Detection Benchmark) for object detection tasks.
- **GTSRB** (German Traffic Sign Recognition Benchmark) for classification.




## 🔍 Tasks & Methods

| Task | Approach |
|------|----------|
| **Detection** | YOLOv5 / YOLOv8 / Faster R-CNN for locating signs in images |
| **Classification** | CNN / Transfer Learning (e.g. ResNet, MobileNet) |
| **Preprocessing** | OpenCV for image manipulation and mask creation |
| **Augmentation** | Rotation, noise, brightness changes, partial occlusion |



## 🧪 Metrics

- **mAP** (mean Average Precision) for detection performance
- **Accuracy**, **Precision**, **Recall**, **F1-score** for classification
- **Inference time** per image (<100 ms)
- **Robustness** to real-world challenges (blur, occlusion, lighting)

