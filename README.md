# ramya_ml_task3
# 🐶🐱 Dogs vs Cats Image Classification

This is my **Task 3** submission for the **Prodigy InfoTech Machine Learning Internship**. The task was to build an image classifier that can tell apart dogs from cats.

---

## 💡 Why I Built It This Way

Most people solving this problem just take a pretrained CNN like VGG16 or ResNet, fine-tune it, and call it done. I wanted to try something different.

Instead of training a deep classifier end-to-end, I used **two CNN models together** — EfficientNetB0 and ResNet50 — purely as feature extractors, and then fed those features into an **SVM classifier**. The idea is that each backbone "sees" the image differently:

- **EfficientNetB0** is great at picking up fine textures — like fur patterns and ear shapes
- **ResNet50** is better at understanding overall structure and depth features

When you combine both, you get a much richer description of the image than either model alone could give. Then instead of using a neural classifier on top, I used an **SVM with an RBF kernel** — which is actually mathematically better at finding clean decision boundaries when your feature space is already well-structured.

I also ran **GridSearchCV** to automatically find the best SVM settings rather than guessing, and used **PCA** to reduce noise in the features before classification.

---

## 🏗️ How the Pipeline Works

```
Image (128×128)
     ↓
EfficientNetB0 ──┐
                  ├──► Concatenate Features
ResNet50      ──┘
     ↓
L2 Normalize → Standard Scale → PCA (100 components)
     ↓
SVM Classifier (RBF Kernel, tuned with GridSearchCV)
     ↓
Cat 🐱 (0)  or  Dog 🐶 (1)
```

---

## 📊 How This Differs from Standard Solutions

| What | Standard Way | What I Did |
|------|-------------|------------|
| Backbone | One CNN (VGG16 / ResNet) | Two CNNs combined (EfficientNet + ResNet50) |
| Classifier | Softmax at the end | SVM with RBF kernel |
| Feature Prep | Raw features | L2 Norm + Scaling + PCA |
| Tuning | Manual | GridSearchCV (automated) |
| Overfitting | Common issue | Greatly reduced |

---

## 🗂️ Dataset

- **From**: [Kaggle Dogs vs Cats](https://www.kaggle.com/c/dogs-vs-cats/data)
- **Training set**: 25,000 images (50% cats, 50% dogs)
- **Test set**: 12,500 images
- **Input size**: All images resized to 128×128

---

## 🛠️ Tools & Libraries Used

- **Python 3.10**
- **TensorFlow / Keras** — for EfficientNetB0 and ResNet50 feature extraction
- **Scikit-learn** — SVM, PCA, GridSearchCV, StandardScaler
- **OpenCV** — image reading and resizing
- **Pandas & NumPy** — data handling
- **Google Colab (T4 GPU)** — where everything was run

---

## ▶️ Running the Notebook

1. Open `TASK3.ipynb` in Google Colab
2. Mount your Google Drive where the dataset is stored
3. Run cells from top to bottom
4. The final cell auto-downloads `submission.csv` to your PC

---

## 📁 What's in This Repo

```
ramya_ml_task3/
├── TASK3.ipynb        → Full code, step by step
├── submission.csv     → Predictions on 1000 test images
└── README.md          → You're reading it!
```

---

## 🔮 What I'd Improve Next

- Visualize predictions using **Grad-CAM** to see what the model actually looks at
- Try adding a **Vision Transformer (ViT)** as a third backbone
- Build a **Streamlit web app** so anyone can upload a photo and get a prediction
- Extend it beyond cats and dogs to classify more animals

---




*Thanks for checking out my project! Feel free to explore the notebook and share feedback.* 🙌
