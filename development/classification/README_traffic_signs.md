# 🧠 Traffic Sign Classification — Technical README

## 1. Overview
This project implements a **Convolutional Neural Network (CNN)** to classify road traffic signs into **43 categories**.  
The model is designed for lightweight yet effective visual recognition and is trained using Keras on an augmented dataset of small RGB images (32×32).

---

## 2. Dataset
**Source Directory:**  
```
/kaggle/input/cvprojectdataset/myData
```

Each subfolder corresponds to a class label (0–42).  
Data is loaded and preprocessed using `ImageDataGenerator`:

```python
ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,      # 80/20 split
    rotation_range=10,
    zoom_range=0.1,
    width_shift_range=0.1,
    height_shift_range=0.1
)
```

**Parameters:**
- Input size: **32×32×3 (RGB)**
- Batch size: **64**
- Data split: **train 80% / validation 20%**

---

## 3. Model Architecture
Sequential CNN defined in `TensorFlow/Keras`:

```text
Input (32x32x3)
 ├─ Conv2D(32, 3x3, ReLU)
 ├─ Conv2D(32, 3x3, ReLU)
 ├─ MaxPooling2D
 ├─ Dropout(0.25)
 ├─ Conv2D(64, 3x3, ReLU)
 ├─ Conv2D(64, 3x3, ReLU)
 ├─ MaxPooling2D
 ├─ Dropout(0.25)
 ├─ Flatten
 ├─ Dense(256, ReLU)
 ├─ Dropout(0.5)
 └─ Dense(43, Softmax)
```

**Compilation:**
```python
model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-3),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
```

This architecture is similar to a simplified **VGG-style** network, optimized for small-scale image data.

---

## 4. Training Pipeline
Model training setup:

```python
model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=10,
    callbacks=[...]
)
```

Typical callbacks:
- `ModelCheckpoint`: saves the best-performing weights
- `EarlyStopping`: halts training on validation stagnation
- `ReduceLROnPlateau`: lowers learning rate when plateauing

---

## 5. Evaluation
After training:

```python
model.evaluate(val_gen)
```

Primary metric: **Accuracy**  
Loss: **Categorical Cross-Entropy**

Expected results — accuracy between **95–99%**, depending on data quality and augmentation parameters.

---

## 6. Inference Example
Load an image and predict its class:

```python
import tensorflow as tf, numpy as np

img = tf.keras.utils.load_img("example.jpg", target_size=(32, 32))
x = tf.keras.utils.img_to_array(img) / 255.0
x = x[None, ...]  # add batch dimension

pred = model.predict(x)
label = np.argmax(pred)
print("Predicted class:", label)
```

---

## 7. Project Structure
```
train_cnn.ipynb       # training notebook
data/
  └── myData/         # dataset folders by class
models/
  └── cnn_traffic.h5  # saved trained model
```

---

## 8. Key Components

| Component | Purpose | Notes |
|------------|----------|-------|
| `ImageDataGenerator` | Normalization + augmentation | improves generalization |
| `Conv2D` layers | Feature extraction | 3×3 filters |
| `Dropout` | Regularization | prevents overfitting |
| `Adam` | Optimizer | adaptive learning rate (1e-3) |
| `Softmax` | Output layer | 43-class probability distribution |

---

## 9. Summary
This CNN provides a balance between performance and computational efficiency for traffic sign recognition tasks.  

