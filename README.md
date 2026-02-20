# ramya_ml_task3
# 🐶🐱 Dogs vs Cats Image Classification

A machine learning project that classifies images of dogs and cats using deep learning feature extraction combined with a Support Vector Machine (SVM) classifier.

## 📌 Project Overview

This project is part of my **Machine Learning Internship at Prodigy InfoTech (Task 3)**.  
The goal is to build a binary image classifier that can distinguish between **dogs (label: 1)** and **cats (label: 0)** using the popular Kaggle Dogs vs Cats dataset.

## 🧠 Model Architecture

- **Feature Extraction**: EfficientNetB0 + ResNet50 (dual backbone, pretrained on ImageNet)
- **Dimensionality Reduction**: PCA (Principal Component Analysis)
- **Classifier**: Support Vector Machine (SVM) with RBF kernel
- **Preprocessing**: L2 Normalization + Standard Scaling

## 📁 Dataset

- **Source**: [Kaggle Dogs vs Cats](https://www.kaggle.com/c/dogs-vs-cats/data)
- **Training images**: 25,000 (12,500 cats + 12,500 dogs)
- **Test images**: 12,500

## 🔧 Tech Stack

| Tool | Purpose |
|------|---------|
| Python | Programming language |
| TensorFlow / Keras | Deep learning feature extraction |
| Scikit-learn | SVM, PCA, GridSearchCV |
| OpenCV | Image loading & preprocessing |
| Google Colab | Development environment (T4 GPU) |
| Pandas / NumPy | Data handling |

## 🚀 How to Run

1. Open `TASK3.ipynb` in Google Colab
2. Mount Google Drive and set dataset path
3. Run all cells in order
4. `submission.csv` will be generated with predictions

## 📊 Results

| Metric | Value |
|--------|-------|
| Model | EfficientNet + ResNet50 + SVM |
| Test Predictions | 1000 images |
| Output Format | `id`, `label` (0 = Cat, 1 = Dog) |

## 📂 Files

```
ramya_ml_task3/
├── TASK3.ipynb          # Main notebook
├── submission.csv       # Test predictions
└── README.md            # Project documentation
```

## 👩‍💻 Author

**Ramya R S**  
AI & ML Engineering Student | Sir MVIT, Bangalore  
GitHub: [@ramyars466](https://github.com/ramyars466)

## 🏢 Internship

**Prodigy InfoTech — Machine Learning Internship**  
Task 3: Image Classification using SVM
