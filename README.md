<div align="center">

# 🧠 Dementia Detection & Severity Prediction
### Deep Learning-Based Classification of Dementia Severity from MRI Scans

[![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)](https://tensorflow.org)
[![Keras](https://img.shields.io/badge/Keras-D00000?style=for-the-badge&logo=keras&logoColor=white)](https://keras.io)
[![Azure ML](https://img.shields.io/badge/Azure%20ML-0078D4?style=for-the-badge&logo=microsoftazure&logoColor=white)](https://azure.microsoft.com/en-us/products/machine-learning)
[![Kaggle](https://img.shields.io/badge/Dataset-Kaggle-20BEFF?style=for-the-badge&logo=kaggle&logoColor=white)](https://www.kaggle.com/datasets/matthewhema/mri-dementia-augmentation-no-data-leak)

[![Best Accuracy](https://img.shields.io/badge/Best%20Accuracy-87%25-brightgreen?style=flat-square)]()
[![Macro F1](https://img.shields.io/badge/Macro%20F1-0.87-brightgreen?style=flat-square)]()
[![Classes](https://img.shields.io/badge/Classes-4-blue?style=flat-square)]()
[![License](https://img.shields.io/badge/License-MIT-lightgrey?style=flat-square)](LICENSE)

<br/>

*Classifying dementia severity into four clinical stages using CNNs, DCGAN-based augmentation,*  
*and Grad-CAM visualizations — achieving **87% accuracy** with interpretable, clinically relevant heatmaps.*

<br/>

[🔍 Overview](#-overview) • [📦 Dataset](#-dataset) • [🏗️ Architecture](#️-architecture) • [📊 Results](#-results) • [🔥 Grad-CAM](#-grad-cam-visualizations) • [🚀 Usage](#-usage) • [📁 Structure](#-project-structure)

</div>

---

## 🔍 Overview

Dementia affects **millions worldwide**, with cases projected to rise significantly in coming decades. Accurate and timely diagnosis is critical for effective clinical intervention — yet traditional diagnostic approaches often lack interpretability and fail to highlight *which* brain regions are driving the prediction.

This project addresses these limitations by combining **Convolutional Neural Networks (CNNs)** with **Grad-CAM (Gradient-weighted Class Activation Mapping)** to:

- 🎯 Classify MRI brain scans into **4 dementia severity levels**
- 🔬 Identify the **specific brain regions** influencing each prediction
- ✅ Generate **visual heatmap explanations** to build trust with medical professionals
- ⚖️ Resolve severe class imbalance using **Deep Convolutional GANs (DCGAN)**

<br/>

### Four Severity Classes

| Label | Class | Clinical Description |
|:---:|:---|:---|
| **Class 0** | 🟢 Non-Demented | No cognitive impairment; normal brain function |
| **Class 1** | 🟡 Very Mild Demented | Slight memory lapses; functionally independent |
| **Class 2** | 🟠 Mild Demented | Noticeable memory loss; difficulty with complex tasks |
| **Class 3** | 🔴 Moderately Demented | Severe impairment; significant loss of independence |

---

## 📦 Dataset

**Source:** [Kaggle — MRI Dementia Dataset](https://www.kaggle.com/datasets/matthewhema/mri-dementia-augmentation-no-data-leak/data)

| Property | Details |
|:---|:---|
| Original Size | 6,400 MRI images across 4 classes |
| Augmented Size | 6,400 images — **1,600 per class** (balanced) |
| Augmentation Method | Deep Convolutional GAN (DCGAN) |
| Image Type | Structural MRI brain scans |
| Classes | Mild, Moderate, Non-Demented, Very Mild Demented |

<br/>

### Why DCGAN Augmentation?

The original dataset suffered from **severe class imbalance** — the Moderate Demented class had far fewer samples than Non-Demented. A naive classifier would simply predict the majority class and still achieve superficially high accuracy while failing clinically.

DCGAN was used to **synthesize realistic MRI images** for underrepresented classes, preserving critical spatial features needed for accurate classification, while bringing every class to **1,600 balanced samples**.

```
Before Augmentation          After DCGAN Augmentation
─────────────────────        ──────────────────────────
Non-Demented    ████████     Non-Demented    ████
Very Mild       ██████       Very Mild       ████
Mild            ███          Mild            ████
Moderate        █            Moderate        ████
                             (Balanced: 1,600 per class)
```

---

## 🏗️ Architecture

### Custom CNN *(Best Performing Model)*

```
Input: MRI Scan
        │
        ▼
┌──────────────────────────────────────────┐
│  Conv Block 1 │ Conv2D → BN → ReLU      │
│               │ MaxPooling → Dropout     │
├──────────────────────────────────────────┤
│  Conv Block 2 │ Conv2D → BN → ReLU      │
│               │ MaxPooling → Dropout     │
├──────────────────────────────────────────┤
│  Conv Block 3 │ Conv2D → BN → ReLU      │
│               │ MaxPooling → Dropout     │
└──────────────────────────────────────────┘
        │
        ▼
  Fully Connected Layers
  Dense → Dropout → Dense → Dropout
        │
        ▼
  Softmax Output (4 classes)
```

**Key design choices:**
- Batch Normalization for stable training
- Dropout regularization to minimize overfitting
- Convolutional layers for spatial feature extraction from MRI slices

<br/>

### DCGAN for Data Augmentation

```
  GENERATOR                           DISCRIMINATOR
  ─────────────────────────────       ─────────────────────────
  Noise Vector (z)                    Real / Synthetic MRI
       ↓                                      ↓
  Dense → Reshape                     Conv → LeakyReLU
       ↓                                      ↓
  TransConv → BN → ReLU              Conv → BN → LeakyReLU
       ↓                                      ↓
  TransConv → BN → ReLU              Conv → BN → LeakyReLU
       ↓                                      ↓
  TransConv → Tanh                    Dense → Sigmoid
  (Synthetic MRI image)               (Real or Fake?)
```

> One DCGAN trained per class — ensuring synthetic images carry class-specific brain anatomy patterns.

<br/>

### Additional Architectures Tested

| Architecture | Type | Notes |
|:---|:---|:---|
| **Custom CNN** | Built from scratch | Best performer — designed for MRI input |
| **DenseNet** | Pre-trained (ImageNet) | Strong generalization via dense connections |
| **InceptionV3** | Pre-trained (ImageNet) | Multi-scale feature extraction |
| **ResNet** | Pre-trained (ImageNet) | Residual connections for deep training |
| **EfficientNet** | Pre-trained (ImageNet) | Struggled with domain adaptation |

---

## 📊 Results

### Model Comparison

| Model | Precision | Recall | F1-Score | Accuracy |
|:---|:---:|:---:|:---:|:---:|
| ⭐ **CNN (Custom)** | **0.87** | **0.87** | **0.87** | **0.87** |
| InceptionV3 | 0.57 | 0.57 | 0.56 | 0.57 |
| DenseNet | 0.55 | 0.55 | 0.55 | 0.55 |
| ResNet | 0.49 | 0.49 | 0.49 | 0.49 |
| EfficientNet | 0.15 | 0.28 | 0.16 | 0.28 |

> 💡 The custom CNN dramatically outperformed all pre-trained transfer learning models, which struggled to adapt from natural ImageNet images to grayscale brain MRI scans.

<br/>

### Key Observations

- ✅ **CNN** achieved 87% accuracy with minimal gap between training and validation — well-generalized
- ✅ **DenseNet** showed moderate performance with stable training curves
- ⚠️ **ResNet** and **InceptionV3** struggled with generalization on MRI data
- ❌ **EfficientNet** showed signs of significant underfitting on this domain
- 🏆 The CNN classified **Very Mild Demented** most accurately (per confusion matrix analysis)

---

## 🔥 Grad-CAM Visualizations

Traditional deep learning models are "black boxes" — high accuracy alone isn't enough for clinical adoption. **Grad-CAM generates heatmaps** that explain *why* the model made a prediction by highlighting the brain regions that most influenced the classification.

```
MRI Input → Forward Pass → Class Score
                ↓
     Backpropagate gradients to
     final convolutional layer
                ↓
     Global average pool → importance weights
                ↓
     Weighted feature map sum → ReLU
                ↓
     Upsample → Overlay on MRI scan
                ↓
     🔵 Low influence  →  🔴 High influence
```

<br/>

### What Grad-CAM Reveals

| Severity | Heatmap Pattern | Clinical Interpretation |
|:---|:---|:---|
| 🟢 Non-Demented | Red areas — minimal activation | Healthy regions with intact structure |
| 🟡 Very Mild | Blue areas highlight key regions | Early subtle changes in medial temporal lobe |
| 🟠 Mild | Focused hippocampal activation | Visible atrophy in memory-related regions |
| 🔴 Moderate | Widespread cortical activation | Global neurodegeneration across multiple lobes |

> ✅ Activations align with known neuroanatomical markers of Alzheimer's progression — confirming the model is learning **real pathological signals**, not imaging artifacts.

**Why this matters for clinicians:**
Rather than receiving a black-box prediction, radiologists can view *which brain regions* drove the classification and verify whether the highlighted structures show expected atrophic changes — enabling true **human-AI collaboration** in diagnosis.

---

## ⚙️ Installation

```bash
# 1. Clone the repository
git clone https://github.com/NK12131/DementiaDetection.git
cd DementiaDetection

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate        # Linux / Mac
# venv\Scripts\activate         # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Download dataset (Kaggle API required)
pip install kaggle
kaggle datasets download -d matthewhema/mri-dementia-augmentation-no-data-leak
unzip mri-dementia-augmentation-no-data-leak.zip -d data/
```

---

## 🚀 Usage

### Train DCGAN *(Balance the dataset first)*
```bash
python train_dcgan.py --class_name ModerateDemented --epochs 300
python train_dcgan.py --class_name MildDemented     --epochs 300
python train_dcgan.py --class_name VeryMildDemented --epochs 300
```

### Train the CNN Classifier
```bash
# Custom CNN (recommended)
python train.py --model cnn --epochs 50 --batch_size 32

# Try alternative architectures
python train.py --model densenet   --pretrained
python train.py --model resnet     --pretrained
python train.py --model inceptionv3 --pretrained
python train.py --model efficientnet --pretrained
```

### Evaluate on Test Set
```bash
python evaluate.py \
  --model_path checkpoints/cnn_best.h5 \
  --data_dir data/test/
```

### Predict + Generate Grad-CAM Heatmap
```bash
python inference.py \
  --image path/to/mri_scan.jpg \
  --model_path checkpoints/cnn_best.h5 \
  --gradcam
```

### Grad-CAM in Python
```python
from gradcam import GradCAM
from tensorflow.keras.models import load_model

model   = load_model('checkpoints/cnn_best.h5')
gradcam = GradCAM(model, layer_name='last_conv_layer')

image   = preprocess_image('scan.jpg')      # → (1, H, W, 1)
heatmap = gradcam.compute(image)
gradcam.overlay(image, heatmap, save_path='results/gradcam_output.png')
```


## 🧩 Challenges & Solutions

| Challenge | Solution |
|:---|:---|
| **Severe class imbalance** (Moderate class severely underrepresented) | DCGAN-based synthetic augmentation to reach 1,600 images per class |
| **GPU resource constraints** during DCGAN training | Resolved using **Azure ML** cloud GPU support |
| **Black-box predictions** limiting clinical trust | Integrated Grad-CAM for visual, region-specific explanations |
| **Transfer learning domain gap** (ImageNet → MRI) | Custom CNN outperformed all pre-trained models |
| **Overfitting vs. underfitting** across architectures | Batch normalization + dropout regularization in custom CNN |

---

## 🏆 Achievements

- ✅ **87% accuracy** classifying dementia into four severity levels
- ✅ **Balanced dataset** via DCGAN — improved generalization across all classes
- ✅ **Grad-CAM interpretability** implemented — highlights clinically meaningful brain regions
- ✅ **Generalized model** with minimal gap between training and validation accuracy
- ✅ **Multi-architecture benchmark** — 5 models systematically compared under identical conditions

---

## 🔭 Future Work

- [ ] **StyleGAN** — Advanced augmentation for higher-fidelity synthetic MRI diversity
- [ ] **Multimodal integration** — Combine MRI with patient history, cognitive scores, and biomarkers
- [ ] **3D Volumetric CNNs** — Process full MRI volumes instead of 2D slices
- [ ] **Other neurodegenerative diseases** — Extend to Parkinson's and Huntington's disease
- [ ] **Federated learning** — Multi-hospital training without sharing patient data
- [ ] **Clinical deployment** — Package as a web-based diagnostic decision-support tool

---

## 📚 References

- [ScienceDirect — Deep Learning in Dementia (Article 1)](https://www.sciencedirect.com/science/article/pii/S1110016822005191)
- [ScienceDirect — Deep Learning in Dementia (Article 2)](https://www.sciencedirect.com)
- [Nature — Neural Networks in Neuroimaging](https://www.nature.com/articles/s41467-022-31037-5)
- [MDPI — CNN Applications in Medical Imaging](https://www.mdpi.com)
- [IEEE — Deep Learning for MRI Analysis](https://ieeexplore.ieee.org/document/9587953)
- Selvaraju et al. (2017). *Grad-CAM: Visual Explanations from Deep Networks.* ICCV.
- Radford et al. (2015). *Unsupervised Representation Learning with DCGANs.* arXiv.

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

<div align="center">

**Dataset:** [Kaggle MRI Dementia](https://www.kaggle.com/datasets/matthewhema/mri-dementia-augmentation-no-data-leak) &nbsp;|&nbsp; **Compute:** [Azure ML](https://azure.microsoft.com/en-us/products/machine-learning)

<br/>

*Built as part of an Applied Data Science research project on explainable AI in clinical neuroimaging.*

<br/>

⭐ **Star this repo if you found it useful!**

</div>
