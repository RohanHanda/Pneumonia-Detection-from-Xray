# 🫁 Chest X-Ray Pneumonia Classification using DenseNet201

This project focuses on **multi-class classification of chest X-ray images** into **Bacterial Pneumonia**, **Viral Pneumonia**, and **Normal** categories using **deep learning and transfer learning**. The model is designed with an emphasis on **medical realism, robustness, and interpretability**.

---

Link to deployed model: [https://huggingface.co/spaces/RohanHanda123/Pneumonia-Detection-using-xray]

---


## 📌 Project Motivation

Chest X-rays are one of the most common and cost-effective diagnostic tools in respiratory medicine. Automated analysis can assist clinicians by:
- Reducing diagnostic workload
- Improving early detection
- Providing decision support in low-resource settings

This project explores whether a CNN-based model can **learn clinically meaningful patterns** from X-ray images and distinguish between visually similar conditions such as **bacterial vs viral pneumonia**.

---

## 🧠 Model Architecture

- **Backbone:** DenseNet201 (pretrained on ImageNet)
- **Input Resolution:** 300 × 300 RGB images
- **Head:**
  - Global Average Pooling
  - Batch Normalization
  - Dense (512 units, ReLU)
  - Dropout (0.5)
  - Dense (3 units, Softmax)

### Why DenseNet201?
- Dense feature reuse improves gradient flow
- Proven effectiveness in medical imaging (e.g., CheXNet)
- Strong performance on texture-based tasks like X-rays

---

## 📂 Dataset

The dataset consists of ~5,000 chest X-ray images organized into three classes:

- **BACTERIAL** (2274 images)
- **VIRAL** (1211 images)
- **NORMAL** (1207 images)

Images were split into **train and validation** sets with consistent class mappings.

---

## 🔧 Preprocessing & Augmentation

Medical-safe preprocessing steps were applied:

- Rescaling pixel values to [0, 1]
- Random rotations (±10°)
- Small width/height shifts
- Zoom augmentation
- Horizontal flips (no vertical flips to preserve anatomy)

These augmentations improve generalization while maintaining clinical validity.

---

## ⚖️ Handling Class Imbalance

The dataset is naturally imbalanced. To address this:

- **Class weights** were computed from the training distribution
- The loss function was weighted accordingly during training

This helped prevent the model from overfitting to the majority (bacterial) class.

---

## 🚀 Training Strategy

Training was performed in **two phases**:

### Phase 1 – Feature Extraction
- DenseNet201 backbone frozen
- Only the custom classification head trained
- Learning rate: 1e-4

### Phase 2 – Fine-Tuning
- Top layers of DenseNet201 unfrozen
- Lower layers kept frozen
- Learning rate reduced to 1e-5

### Callbacks Used
- ModelCheckpoint (best validation AUC)
- EarlyStopping
- ReduceLROnPlateau

This staged approach ensures stable convergence and better domain adaptation.

---

## 📊 Final Results

### 🔹 Overall Metrics
- **Accuracy:** 84%
- **Macro F1-score:** 0.83
- **AUC:** 0.94

### 🔹 Class-wise Performance

| Class       | Precision | Recall | F1-score |
|------------|----------|--------|----------|
| Bacterial  | 0.86     | 0.88   | 0.87     |
| Normal     | 0.86     | 0.97   | 0.91     |
| Viral      | 0.77     | 0.66   | 0.71     |

---
## 🔍 Model Interpretability (Grad-CAM)

To improve transparency and trust, **Grad-CAM** was used to visualize model attention.

- Highlights lung regions contributing most to predictions
- Confirms that the model focuses on anatomically relevant areas

This is especially important for medical AI applications.

---

## 🛠️ Tech Stack

- Python
- TensorFlow / Keras
- NumPy, OpenCV
- Scikit-learn
- Matplotlib
- Gradio (for interactive demo)

---

## 📈 Key Learnings

- Transfer learning significantly outperforms training from scratch
- DenseNet architectures are well-suited for medical imaging
- AUC and macro F1 are more reliable than accuracy alone for imbalanced medical datasets
- Interpretability is crucial when working with healthcare data

---

## 🚧 Limitations & Future Work

- Viral vs bacterial pneumonia remains challenging due to visual overlap
- Performance may improve with:
  - Larger datasets
  - Medical-domain pretraining (RadImageNet, CheXNet weights)
  - Two-stage classification pipelines
  - Model ensembling

---

## 📜 Disclaimer

This project is for **educational and research purposes only** and is **not intended for clinical use**.

---
