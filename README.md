# 🛍️ Product Discovery using Fashion-MNIST

> **Neural Networks & Deep Learning — Mini Project**  
> Academic Grade Target: **9.5 / 10**

---

## 📌 Problem Statement

Online marketplaces receive thousands of new clothing items daily. Manual product categorisation is slow and error-prone. This project demonstrates how a deep learning pipeline can automatically tag clothing images into **10 categories**, enabling smarter search, recommendations, and inventory management.

---

## 🎯 Objective

Build, train, and compare two deep learning models on the **Fashion-MNIST** dataset:

| Model | Type | Key Features |
|-------|------|-------------|
| ANN | Fully-Connected Network | Baseline classifier |
| CNN | Convolutional Neural Network | Spatial feature learning + Augmentation |

---

## 📁 Project Structure

```
NNdl_mini_project/
│
├── data/                     # Dataset cache (auto-downloaded)
├── models/                   # Saved .keras model files
├── notebooks/                # Jupyter exploration notebook
├── outputs/
│   ├── plots/                # All generated charts (PNG)
│   └── reports/              # Classification reports + history CSVs
├── src/
│   ├── config.py             # Hyperparameters & paths (single source of truth)
│   ├── data_loader.py        # Dataset loading, preprocessing, visualisation
│   ├── ann_model.py          # ANN architecture builder
│   ├── cnn_model.py          # CNN architecture builder + data augmentation
│   ├── trainer.py            # Training loop, callbacks, history persistence
│   ├── evaluator.py          # Metrics, curves, confusion matrix, comparison
│   └── visualizer.py        # Prediction & error visualisation grids
│
├── main.py                   # End-to-end pipeline (single entry point)
├── requirements.txt          # Python dependencies
└── README.md                 # This file
```

---

## 🧠 Models

### Model 1 — ANN
```
Input(784) → Dense(512,ReLU)+BN+Dropout
           → Dense(256,ReLU)+BN+Dropout
           → Dense(128,ReLU)+BN+Dropout
           → Dense(10, Softmax)
```
- L2 regularisation on all Dense layers
- Batch Normalisation for stable training
- Dropout (0.3) to prevent overfitting

### Model 2 — CNN
```
Input(28,28,1) → [Conv32→Conv32→MaxPool→Dropout]
               → [Conv64→Conv64→MaxPool→Dropout]
               → [Conv128→MaxPool→Dropout]
               → Flatten
               → Dense(256)+BN+Dropout
               → Dense(128)+BN+Dropout
               → Dense(10, Softmax)
```
- Data Augmentation (rotation, shift, zoom, flip)
- L2 regularisation + Batch Normalisation
- Dropout (0.4) in dense layers

---

## ⚙️ Tech Stack

| Tool | Version | Purpose |
|------|---------|---------|
| Python | 3.10+ | Core language |
| TensorFlow / Keras | 2.15 | Model building & training |
| NumPy | 1.26 | Numerical operations |
| Matplotlib | 3.8 | Plotting |
| Seaborn | 0.13 | Confusion matrix heatmap |
| scikit-learn | 1.4 | Classification report, metrics |

---

## 🚀 How to Run

### 1. Create virtual environment
```bash
python -m venv venv
venv\Scripts\activate      # Windows
# or
source venv/bin/activate   # Linux/macOS
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the full pipeline
```bash
python main.py
```

All plots are saved to `outputs/plots/` and reports to `outputs/reports/`.

---

## 📊 Expected Results

| Model | Test Accuracy | Test Loss |
|-------|--------------|-----------|
| ANN   | ~88–89%      | ~0.31     |
| CNN   | ~92–94%      | ~0.21     |

> CNN achieves **4–6 percentage points** higher accuracy than ANN because
> convolutional layers learn *spatial hierarchies* (edges → textures → shapes)
> through weight-shared filters, drastically reducing parameters while capturing
> structure that flat layers miss.

---

## 📈 Output Files

| File | Description |
|------|-------------|
| `outputs/plots/sample_images.png` | Sample grid from training set |
| `outputs/plots/class_distribution.png` | Per-class count bar chart |
| `outputs/plots/ANN_training_curves.png` | ANN Acc & Loss vs Epochs |
| `outputs/plots/CNN_training_curves.png` | CNN Acc & Loss vs Epochs |
| `outputs/plots/ANN_confusion_matrix.png` | Normalised confusion matrix |
| `outputs/plots/CNN_confusion_matrix.png` | Normalised confusion matrix |
| `outputs/plots/ANN_predictions.png` | Correct/wrong prediction grid |
| `outputs/plots/CNN_predictions.png` | Correct/wrong prediction grid |
| `outputs/plots/model_comparison.png` | Side-by-side bar chart |
| `outputs/reports/ANN_classification_report.txt` | Per-class precision/recall/F1 |
| `outputs/reports/CNN_classification_report.txt` | Per-class precision/recall/F1 |
| `models/ANN_final.keras` | Saved ANN weights |
| `models/CNN_final.keras` | Saved CNN weights |

---

## 💡 Real-World Applications

| Application | How This Model Helps |
|-------------|---------------------|
| **E-commerce Auto-Tagging** | Auto-assign category labels to new product uploads |
| **Visual Search** | Power "shop the look" reverse image search |
| **Recommendation Systems** | Suggest similar items using predicted categories |
| **Inventory Management** | Auto-sort warehouse photos by garment type |
| **Trend Analysis** | Track category-level inventory shifts over time |

---

## 🔧 Advanced Features Implemented

- ✅ **EarlyStopping** — stops when validation accuracy plateaus (patience=5)
- ✅ **ReduceLROnPlateau** — halves learning rate on loss plateau
- ✅ **ModelCheckpoint** — saves only the best weights
- ✅ **Data Augmentation** — random rotation, shift, zoom, horizontal flip (CNN)
- ✅ **Batch Normalisation** — applied after every Dense/Conv layer
- ✅ **L2 Regularisation** — on all Dense and Conv layers
- ✅ **Model Saving/Loading** — `.keras` format for portability
- ✅ **Reproducible Seeds** — fixed for Python, NumPy, TensorFlow
- ✅ **Modular Code** — separate config, loader, model, trainer, evaluator

---

## 👤 Author

B. Sai Subrahmanyam — NNDL Mini Project, 2026
