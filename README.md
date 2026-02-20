# ramya_ml_task3
# 🐶🐱 Dogs vs Cats Image Classification
### Prodigy InfoTech — Machine Learning Internship | Task 3

[![Python](https://img.shields.io/badge/Python-3.10-blue)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)](https://tensorflow.org)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.x-green)](https://scikit-learn.org)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

---

## 📌 Project Overview

This project builds a **binary image classifier** to distinguish between **dogs (1)** and **cats (0)** using a novel hybrid deep learning + classical ML pipeline. It goes beyond standard CNN-only approaches by combining **dual backbone feature extraction** with an **optimized SVM classifier**.

---

## 🚀 What Makes Our Solution Novel?

Most existing solutions for Dogs vs Cats classification use a single CNN model end-to-end (e.g., just VGG16 or just ResNet). Our approach is different in several key ways:

### 1. 🔀 Dual Backbone Feature Fusion
> Standard approach: Single CNN (e.g., VGG16 only)  
> **Our approach: EfficientNetB0 + ResNet50 combined**

- **EfficientNetB0** captures fine-grained texture details efficiently
- **ResNet50** captures deep structural and shape features
- Both feature vectors are **concatenated** → richer representation than any single model
- This gives **complementary features** that a single backbone misses

### 2. 🤖 Hybrid Deep Learning + Classical ML
> Standard approach: CNN → Softmax (end-to-end deep learning)  
> **Our approach: CNN Features → PCA → SVM (hybrid pipeline)**

- SVMs with **RBF kernel** are mathematically proven to find optimal decision boundaries
- Avoids overfitting that deep classifiers often face on medium-sized datasets
- Requires **less training data** to generalize compared to full fine-tuning

### 3. 📉 Intelligent Dimensionality Reduction
> Standard approach: Raw feature vectors fed directly to classifier  
> **Our approach: L2 Normalization → Standard Scaling → PCA**

- L2 normalization ensures both backbones contribute equally
- PCA removes noise and redundancy from high-dimensional feature space
- Reduces computation while retaining maximum variance

### 4. ⚙️ Automated Hyperparameter Optimization
> Standard approach: Manual tuning of C, gamma  
> **Our approach: GridSearchCV with cross-validation**

- Automatically finds best `C` and `gamma` values for SVM
- 5-fold cross-validation ensures robust, unbiased performance

---

## 🧠 Model Architecture

```
Input Image (128×128×3)
        ↓
┌───────────────────────────────────┐
│   Dual Backbone Feature Extraction │
│  ┌─────────────┐ ┌─────────────┐  │
│  │ EfficientNet│ │   ResNet50  │  │
│  │    B0       │ │  (pretrained│  │
│  │ (pretrained)│ │  ImageNet)  │  │
│  └──────┬──────┘ └──────┬──────┘  │
│         └───────┬────────┘         │
│           Concatenate              │
└───────────────────────────────────┘
        ↓
   L2 Normalization
        ↓
   Standard Scaling
        ↓
   PCA (n_components=100)
        ↓
   SVM (RBF Kernel)
   [GridSearchCV Optimized]
        ↓
   Output: Cat (0) / Dog (1)
```

---

## 📁 Dataset

- **Source**: [Kaggle Dogs vs Cats](https://www.kaggle.com/c/dogs-vs-cats/data)
- **Training images**: 25,000 (12,500 cats + 12,500 dogs)
- **Test images**: 12,500
- **Image size**: Resized to 128×128 pixels

---

## 🔧 Tech Stack

| Tool | Version | Purpose |
|------|---------|---------|
| Python | 3.10 | Programming language |
| TensorFlow / Keras | 2.x | Feature extraction (EfficientNet, ResNet) |
| Scikit-learn | 1.x | SVM, PCA, GridSearchCV, Scaler |
| OpenCV | 4.x | Image loading & preprocessing |
| NumPy | Latest | Array operations |
| Pandas | Latest | CSV handling |
| Google Colab | T4 GPU | Development environment |

---

## 📊 Results

| Component | Detail |
|-----------|--------|
| Backbone 1 | EfficientNetB0 (pretrained ImageNet) |
| Backbone 2 | ResNet50 (pretrained ImageNet) |
| Reduction | PCA → 100 components |
| Classifier | SVM RBF kernel (GridSearchCV tuned) |
| Test Predictions | 1000 images |
| Output | `id`, `label` (0 = Cat, 1 = Dog) |

---

## 🆚 Comparison with Existing Solutions

| Feature | Standard CNN Approach | **Our Hybrid Approach** |
|---|---|---|
| Backbone | Single (e.g., VGG16) | **Dual (EfficientNet + ResNet50)** |
| Classifier | Softmax layer | **SVM (RBF kernel)** |
| Feature Normalization | None / BatchNorm | **L2 + Standard Scaling** |
| Dimensionality Reduction | None | **PCA** |
| Hyperparameter Tuning | Manual | **GridSearchCV (auto)** |
| Overfitting Risk | High | **Low** |
| Training Data Needed | Large | **Moderate** |

---

## 🚀 How to Run

### Prerequisites
```bash
pip install tensorflow scikit-learn opencv-python pandas numpy
```

### Steps
1. Open `TASK3.ipynb` in **Google Colab**
2. Mount Google Drive:
```python
from google.drive import drive
drive.mount('/content/drive')
```
3. Set dataset path and run all cells in order
4. `submission.csv` will be auto-generated and downloaded

---

## 📂 Repository Structure

```
ramya_ml_task3/
├── TASK3.ipynb          # Main Jupyter notebook (full pipeline)
├── submission.csv       # Final test predictions (1000 images)
└── README.md            # Project documentation
```

---

## 🔮 Future Improvements

- [ ] Add Grad-CAM visualizations to explain predictions
- [ ] Try Vision Transformer (ViT) as third backbone
- [ ] Deploy as a web app using Streamlit
- [ ] Extend to multi-class pet classification

---

This project is licensed under the **MIT License** — free to use and modify.

