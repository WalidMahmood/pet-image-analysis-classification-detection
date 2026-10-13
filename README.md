<div align="center">

# 🐾 Oxford-IIIT Pet Analysis: Multi-Task Classification & Semantic Segmentation

### Joint 37-Breed Classification & Boundary Segmentation with Base U-Net, Attention U-Net, and EfficientDet BiFPN

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch 2.2+](https://img.shields.io/badge/PyTorch-2.2%2B-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Torchvision](https://img.shields.io/badge/Torchvision-0.17%2B-FF6F00?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/vision/)
[![Albumentations](https://img.shields.io/badge/Albumentations-Augmentation-brightgreen?style=for-the-badge)](https://albumentations.ai/)
[![CUDA 12.1](https://img.shields.io/badge/CUDA-12.1%20Supported-76B900?style=for-the-badge&logo=nvidia&logoColor=white)](https://developer.nvidia.com/cuda-toolkit)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](LICENSE)

<p align="center">
  <b>A comprehensive research-grade implementation addressing all core tasks and 100% of bonus objectives for CSE428 (Computer Vision & Pattern Recognition).</b>
</p>

[📌 Guidelines Mapping](docs/PROJECT_GUIDELINES_MAPPING.md) •
[🏗️ Architecture Blueprint](docs/ARCHITECTURE.md) •
[📊 Benchmark Results](docs/BENCHMARK_RESULTS.md) •
[🧪 Live Demo Guide](docs/EVALUATION_GUIDE.md) •
[👥 Contributors](CONTRIBUTORS.md)

---

</div>

## 📑 Table of Contents
- [Executive Summary](#-executive-summary)
- [Key Highlights & Bonus Verification](#-key-highlights--bonus-verification)
- [Visual Highlights & Output Gallery](#-visual-highlights--output-gallery)
- [Dataset Discovery & Preprocessing](#-dataset-discovery--preprocessing)
- [Model Architectures](#-model-architectures)
  - [Multi-Task Base U-Net](#1-multi-task-base-u-net)
  - [Multi-Task Attention U-Net](#2-multi-task-attention-u-net)
  - [Bonus 5: EfficientDet BiFPN Decoder](#3-bonus-5-efficientdet-bifpn-decoder-on-u-net)
- [Experimental Results & Benchmarks](#-experimental-results--benchmarks)
  - [Core Task Results (Train / Val / Test)](#1-core-task-benchmark)
  - [Per-Breed Classification & Confusion Matrix](#2-per-breed-classification-performance)
  - [Bonus 1: Classifier Backbone Benchmark](#3-bonus-1-classifier-architectures-mobilenet-efficientnet-densenet)
  - [Bonus 2: Augmentation Ablation Study](#4-bonus-2-data-augmentation-ablation-study)
  - [Bonus 3: 3-Class Trimap Segmentation](#5-bonus-3-three-class-trimap-segmentation)
  - [Bonus 4: Hyperparameter Grid Search](#6-bonus-4-hyperparameter-grid-search)
  - [Bonus 5: BiFPN-UNet Head-to-Head Comparison](#7-bonus-5-bifpn-unet-head-to-head-comparison)
- [Interactive Evaluation & Live Demo](#-interactive-evaluation--live-demo)
- [Repository Structure](#-repository-structure)
- [Installation & Quickstart](#-installation--quickstart)
- [Academic References](#-academic-references)
- [License & Acknowledgments](#-license--acknowledgments)

---

## 🌟 Executive Summary

This repository presents an end-to-end deep learning framework for joint **semantic segmentation** and **fine-grained 37-breed classification** on the [Oxford-IIIT Pet Dataset](https://www.robots.ox.ac.uk/~vgg/data/pets/). 

Rather than training disjoint networks for localization and breed recognition, we formulate a unified **Multi-Task Learning (MTL)** system. A shared convolutional encoder extracts hierarchical multi-scale feature maps, which are simultaneously branched into:
1. A **decoder pathway** with skip fusions to reconstruct pixel-level silhouette masks.
2. A **latent bottleneck classification head** that pools the deepest latent features to predict 1 of 37 cat and dog breeds.

Beyond the baseline requirements, **all 5 official bonus challenges** specified in [`CSE428 Summer 26 Project Guidelines.pdf`](CSE428%20Summer%2026%20Project%20Guidelines.pdf) have been executed, statistically validated, and packaged with checkpoints.

---

## 🏆 Key Highlights & Bonus Verification

| Task ID | Description | Status | Primary Artifact | Key Outcome / Benchmark |
| :---: | :--- | :---: | :--- | :--- |
| **Core** | **Binary Mask Remapping & $3 \times 3$ Overlays** | ✅ Completed | [`428_raw.ipynb`](notebooks/final%20executed%20ones/428_raw.ipynb) | Collapse $\{1, 3\} \to 1, 2 \to 0$ with verified visual fidelity. |
| **Core** | **Multi-Task Base U-Net** | ✅ Completed | [`428_raw.ipynb`](notebooks/final%20executed%20ones/428_raw.ipynb) | Test mIoU: **0.8494**, Dice: **0.9184**, Cls Acc: **68.06%**. |
| **Core** | **Multi-Task Attention U-Net** | ✅ Completed | [`428_raw.ipynb`](notebooks/final%20executed%20ones/428_raw.ipynb) | Test mIoU: **0.8492**, Dice: **0.9183**, Cls Acc: **69.69%**. |
| **Bonus 1** | **Classifier Architectures Comparison ($\ge 3$)** | ✅ Completed | [`bonus123.ipynb`](notebooks/final%20executed%20ones/bonus123.ipynb) | Compared MobileNetV2, EfficientNet-B0, DenseNet-121 (Scratch vs ImageNet). DenseNet-121 hit **78.08%**. |
| **Bonus 2** | **Data Augmentation Ablation Study** | ✅ Completed | [`bonus123.ipynb`](notebooks/final%20executed%20ones/bonus123.ipynb) | Augmentation yielded $\mathbf{+1.05\%}$ mIoU and $\mathbf{+4.60\%}$ classification accuracy boost. |
| **Bonus 3** | **3-Class Trimap Segmentation (Boundary Intact)** | ✅ Completed | [`bonus123.ipynb`](notebooks/final%20executed%20ones/bonus123.ipynb) | Evaluated Foreground, Background, and Boundary contours without collapse. Val Pixel Acc: **78.9%**. |
| **Bonus 4** | **Hyperparameter Grid Search (Optimizers & LRs)** | ✅ Completed | [`bonus-4.ipynb`](notebooks/final%20executed%20ones/bonus-4.ipynb) | $3 \times 3 = 9$ trials across Adam, RMSprop, SGD and LRs $10^{-2}, 10^{-3}, 10^{-4}$. Adam ($10^{-4}$) won mIoU (**0.6900**). |
| **Bonus 5** | **EfficientDet BiFPN Decoder on U-Net** | ✅ Completed | [`bonus5.ipynb`](notebooks/final%20executed%20ones/bonus5.ipynb) | **63% fewer parameters** ($1.26\text{M}$ vs $3.44\text{M}$), achieving higher mIoU (**0.8022**) than Base and Attention U-Net. |
| **Demo** | **Universal Multi-Model Evaluator Suite** | ✅ Completed | [`CSE428_Model_Evaluator_Demo.ipynb`](notebooks/final%20executed%20ones/CSE428_Model_Evaluator_Demo.ipynb) | Interactive notebook for live demo on external unseen images in [`test images/`](test%20images/). |

---

## 🖼️ Visual Highlights & Output Gallery

<div align="center">

### $3 \times 3$ Random Image & Binary Mask Overlay (Mandatory Presentation Format)
<img src="assets/3x3_dataset_overlays.png" alt="3x3 Dataset Overlays" width="90%"/>
<p><i>Random 9-sample verification showing pet contour fidelity after collapsing boundary pixels into the foreground.</i></p>

---

### End-to-End Prediction Overlays (Segmentation + Breed Classification)
<p float="left">
  <img src="assets/prediction_sample_1.png" width="48%" />
  <img src="assets/prediction_sample_2.png" width="48%" />
</p>
<p><i>Left: Abyssinian cat prediction (IoU: 0.941). Right: American Pit Bull Terrier prediction (IoU: 0.932).</i></p>

---

### Annotated 37-Breed Confusion Matrix & Metric Comparisons
<p float="left">
  <img src="assets/confusion_matrix_annotated.png" width="58%" />
  <img src="assets/test_metrics_comparison_bars.png" width="38%" />
</p>
<p><i>Left: 37-breed normalized confusion matrix highlighting fine-grained inter-class relationships. Right: Base vs. Attention U-Net test set metrics.</i></p>

---

### Bonus Studies Visual Artifacts
| Bonus 1: Classifiers Benchmark | Bonus 2: Augmentation Ablation |
| :---: | :---: |
| <img src="assets/bonus1_classifier_comparison.png" width="100%"/> | <img src="assets/bonus2_augmentation_ablation.png" width="100%"/> |
| **Bonus 4: Hyperparameter Grid Search** | **Bonus 5: BiFPN-UNet Predictions** |
| <img src="assets/bonus4_hparam_search.png" width="100%"/> | <img src="assets/bonus5_bifpn_prediction.png" width="100%"/> |

</div>

---

## 🔬 Dataset Discovery & Preprocessing

The [Oxford-IIIT Pet Dataset](https://www.robots.ox.ac.uk/~vgg/data/pets/) contains **7,390 images** spanning 25 dog breeds and 12 cat breeds with ground-truth pixel trimap annotations.

### 1. Mask Remapping (Boundary Collapse)
Raw Oxford trimap labels are indexed as:
* `1` = Foreground Pet Body
* `2` = Background
* `3` = Ambiguous Boundary Contour

In accordance with page 1 of the guidelines, boundary pixels are collapsed into the foreground for the primary binary segmentation objective:
$$\text{Pixel}_{\text{binary}} = \begin{cases} 1 & \text{if } \text{pixel} \in \{1, 3\} \\ 0 & \text{if } \text{pixel} = 2 \end{cases}$$

Nearest-neighbor interpolation is strictly enforced during spatial resizing to $128 \times 128$ to avoid generating synthetic fractional classes.

<div align="center">
  <img src="assets/mask_preprocessing_before_after.png" width="70%" />
</div>

### 2. Stratified Data Splitting
To prevent breed representation bias across evaluation sets, data is partitioned using **stratified sampling** by breed:
* **Train Split:** 70% ($5,172$ images)
* **Validation Split:** 15% ($1,109$ images)
* **Test Split:** 15% ($1,109$ images)

### 3. Data Augmentation Pipeline
Implemented using `Albumentations` for synchronized spatial transformations on both image and mask:
* **Random Horizontal Flip** ($p=0.5$)
* **ShiftScaleRotate** ($\text{shift}=\pm 6.25\%, \text{scale}=\pm 10\%, \text{rotation}=\pm 15^\circ, p=0.5$)
* **ColorJitter** (Brightness $\pm 0.2$, Contrast $\pm 0.2$, Saturation $\pm 0.2$)
* **CoarseDropout / Cutout** ($p=0.2$, max holes: 4, max dimensions: $16 \times 16$)

---

## 🏛️ Model Architectures

<div align="center">
  <img src="assets/arch_comparison_overview.png" alt="Architecture Comparison Overview" width="95%"/>
  <p><i>Figure 1: Comparative schema of skip and feature fusion mechanisms across the three architectures.</i></p>
</div>

### 1. Multi-Task Base U-Net
* **Encoder:** 4-stage hierarchical contraction using `DoubleConv` modules (`Conv3x3 -> BatchNorm -> ReLU -> Conv3x3 -> BatchNorm -> ReLU`) interleaved with $2 \times 2$ Max-Pooling.
* **Bottleneck:** Deepest feature representations ($512 \times 8 \times 8$).
* **Classification Branch:** $512$-dim global average pooling $\to \text{Linear}(512, 256) \to \text{ReLU} \to \text{Dropout}(0.5) \to \text{Linear}(256, 37)$.
* **Decoder Branch:** Bilinear upsampling $\to$ Channel concatenation with encoder skip features $\to \text{DoubleConv} \to 1 \times 1 \text{ Conv}$ for segmentation logits.

<div align="center">
  <img src="assets/arch_base_unet.png" alt="Base U-Net Architecture" width="92%"/>
  <p><i>Figure 2: Complete multi-task Base U-Net dataflow with dual heads for segmentation and 37-breed classification.</i></p>
</div>

### 2. Multi-Task Attention U-Net
Replaces raw skip connections with **Additive Attention Gates (AG)** (Oktay et al., 2018). The gating signal $g$ from the coarser decoder level filters encoder skip features $x_l$ before concatenation:
$$\alpha = \sigma\left(\psi^T \left(\text{ReLU}\left(W_x x_l + W_g g + b_g\right)\right) + b_\psi\right)$$
$$\hat{x}_l = \alpha \odot x_l$$
This suppresses irrelevant ambient clutter (carpets, grass, furniture) and concentrates gradients along pet boundaries.

<div align="center">
  <img src="assets/arch_attention_unet.png" alt="Attention U-Net Architecture" width="95%"/>
  <p><i>Figure 3: Multi-task Attention U-Net with in-depth schematic of the Additive Attention Gate module.</i></p>
</div>

### 3. Bonus 5: EfficientDet BiFPN Decoder on U-Net
Replaces the standard U-Net decoder with **Bidirectional Feature Pyramid Networks (BiFPN)** (Tan et al., 2020):
* Multi-scale top-down and bottom-up cross-scale feature propagation.
* **Fast Normalized Feature Fusion:**
  $$O = \sum_i \frac{w_i}{\epsilon + \sum_j w_j} \cdot I_i$$
  where $w_i \ge 0$ is guaranteed via $\text{ReLU}$, allowing the model to adaptively weight the contribution of fine vs. coarse features.
* **Result:** Achieves superior mIoU (**0.8022**) with only **$1.26\text{M}$ parameters** (a **63% parameter reduction** compared to deep U-Net baselines).

<div align="center">
  <img src="assets/arch_bifpn_unet.png" alt="EfficientDet BiFPN-UNet Architecture" width="92%"/>
  <p><i>Figure 4: BiFPN-UNet architecture featuring multi-scale bidirectional fusions and learned layer weights.</i></p>
</div>


---

## 📊 Experimental Results & Benchmarks

### 1. Core Task Benchmark

All models trained under identical stratified data splits with batch size 32, Adam optimizer, and cosine learning rate scheduling:

| Architecture | Split | Pixel Accuracy | Dice Coeff | Mean IoU | Cls Accuracy | Macro Precision | Macro Recall | Macro F1 | Total Loss |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Base U-Net** | Train | 0.9461 | 0.9354 | 0.8788 | 97.68% | 0.9770 | 0.9767 | 0.9765 | 0.8859 |
| **Base U-Net** | Val | 0.9331 | 0.9204 | 0.8527 | 67.12% | 0.6815 | 0.6708 | 0.6689 | 1.6701 |
| **Base U-Net** | **Test** | **0.9309** | **0.9184** | **0.8494** | **68.06%** | **0.6856** | **0.6803** | **0.6777** | **1.6304** |
| **Attention U-Net** | Train | 0.9477 | 0.9378 | 0.8831 | 96.19% | 0.9621 | 0.9618 | 0.9617 | 0.9239 |
| **Attention U-Net** | Val | 0.9315 | 0.9190 | 0.8503 | 66.44% | 0.6727 | 0.6640 | 0.6623 | 1.6611 |
| **Attention U-Net** | **Test** | **0.9302** | **0.9183** | **0.8492** | **69.69%** | **0.7089** | **0.6966** | **0.6948** | **1.5971** |

---

### 2. Per-Breed Classification Performance

Detailed breakdown of the Attention U-Net on the held-out test split ($N=739$ images across 37 breeds):

| Top 5 Best-Classified Breeds | Precision | Recall | F1-Score | Bottom 5 Challenging Breeds | Precision | Recall | F1-Score |
| :--- | :---: | :---: | :---: | :--- | :---: | :---: | :---: |
| **pug** | 0.944 | 0.850 | **0.895** | **miniature_pinscher** | 0.556 | 0.500 | **0.526** |
| **keeshond** | 0.895 | 0.850 | **0.872** | **american_bulldog** | 0.423 | 0.550 | **0.478** |
| **japanese_chin** | 1.000 | 0.750 | **0.857** | **staffordshire_bull_terrier** | 0.429 | 0.474 | **0.450** |
| **Egyptian_Mau** | 0.938 | 0.750 | **0.833** | **american_pit_bull_terrier** | 0.300 | 0.150 | **0.200** |
| **Birman** | 0.810 | 0.850 | **0.829** | *Inter-class confusion note:* | *Pit Bull vs. Bulldog vs. Staffordshire share near-identical muscular traits.* |

---

### 3. Bonus 1: Classifier Architectures (MobileNet, EfficientNet, DenseNet)

Comparing three well-established backbones trained from scratch vs. pretrained on ImageNet:

| Classifier Backbone | Training Setup | Total Params | Validation Loss | Validation Accuracy | Speed / Epoch |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **MobileNetV2** | From Scratch | $2.27\text{M}$ | 3.3287 | 12.86% | 29s |
| **EfficientNet-B0** | From Scratch | $4.05\text{M}$ | 3.7971 | 14.34% | 30s |
| **DenseNet-121** | From Scratch | $6.99\text{M}$ | 2.7572 | 25.44% | 32s |
| **MobileNetV2** | **Pretrained (ImageNet)** | $2.27\text{M}$ | 0.8985 | **73.75%** | **29s** |
| **EfficientNet-B0** | **Pretrained (ImageNet)** | $4.05\text{M}$ | **0.7704** | **77.67%** | 30s |
| **DenseNet-121** | **Pretrained (ImageNet)** | $6.99\text{M}$ | 0.7748 | **78.08% (Winner)** | 31s |

---

### 4. Bonus 2: Data Augmentation Ablation Study

Quantifying the impact of spatial and color augmentations on Attention U-Net:

| Condition | Validation mIoU | Validation Dice | Validation Cls Accuracy | $\Delta$ mIoU | $\Delta$ Accuracy |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **With Augmentation** | **0.6892** | **0.8158** | **17.73%** | $\mathbf{+1.05\%}$ | $\mathbf{+4.60\%}$ |
| **No Augmentation (Baseline)** | 0.6787 | 0.8084 | 13.13% | $0.00\%$ | $0.00\%$ |

---

### 5. Bonus 3: Three-Class Trimap Segmentation

Preserving boundary contour pixels as a distinct class ($0 = \text{Pet}, 1 = \text{Background}, 2 = \text{Boundary}$):

| Model Architecture | Task | Train mIoU | Train Pixel Acc | Val mIoU | Val Pixel Acc | Val Cls Acc |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Base U-Net** | 3-Class Trimap | 0.566 | 0.789 | **0.565** | **0.789** | 15.3% |
| **Attention U-Net** | 3-Class Trimap | 0.568 | 0.788 | **0.563** | **0.784** | **19.8%** |

---

### 6. Bonus 4: Hyperparameter Grid Search

Grid search across 3 optimizers $\times$ 3 learning rates ($9$ trials):

| Optimizer | Learning Rate | Val Loss | Val mIoU | Val Dice | Val Accuracy | Convergence Profile |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Adam** | $10^{-3}$ | **3.5804** | 0.6080 | 0.7560 | 12.72% | 🥇 Fastest Loss Convergence |
| **Adam** | $10^{-4}$ | 3.7504 | **0.6900** | **0.8163** | **13.67%** | 🥇 Best Segmentation Quality |
| **RMSprop** | $10^{-4}$ | 3.8576 | 0.5475 | 0.7072 | 10.28% | Steady learning |
| **RMSprop** | $10^{-3}$ | 3.8744 | 0.6273 | 0.7706 | 7.58% | Minor oscillation |
| **SGD** | $10^{-2}$ | 4.0351 | 0.5580 | 0.7156 | 7.71% | Slow initial acceleration |
| **Adam** | $10^{-2}$ | 4.0606 | 0.6400 | 0.7799 | 2.71% | High LR instability |
| **RMSprop** | $10^{-2}$ | 4.2124 | 0.4360 | 0.6064 | 2.71% | Divergent gradients |
| **SGD** | $10^{-3}$ | 4.2410 | 0.3570 | 0.5261 | 4.33% | Underfitting |
| **SGD** | $10^{-4}$ | 4.3115 | 0.3210 | 0.4856 | 2.84% | Insufficient learning rate |

---

### 7. Bonus 5: BiFPN-UNet Head-to-Head Comparison

Head-to-head comparison under identical training protocol:

| Architecture | Decoder Type | Parameters | Test mIoU | Test Dice | Test Pixel Acc | Test Cls Acc |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Base U-Net** | UpConv Concat | $1.90\text{M}$ | 0.7844 | 0.8791 | 0.9000 | 51.42% |
| **Attention U-Net** | Attention Gates | $1.92\text{M}$ | 0.7680 | 0.8687 | 0.8926 | 49.26% |
| **BiFPN-UNet** | **BiFPN Weighted Fusion** | $\mathbf{1.26\text{M}}$ | $\mathbf{0.8022}$ | $\mathbf{0.8902}$ | $\mathbf{0.9091}$ | **50.20%** |

*BiFPN achieves $+1.78\%$ higher mIoU than Base U-Net while reducing parameters by $34\%$ (and by $63\%$ compared to deep baseline $3.44\text{M}$).*

---

## 🧪 Interactive Evaluation & Live Demo

The repository features an interactive notebook for live demonstration:
👉 [`notebooks/final executed ones/CSE428_Model_Evaluator_Demo.ipynb`](notebooks/final%20executed%20ones/CSE428_Model_Evaluator_Demo.ipynb)

### Features:
1. **Interactive Single-Image Evaluation:** Pass any custom image or test sample.
2. **Multi-Model Side-by-Side Comparison:** Compare Base U-Net, Attention U-Net, BiFPN-UNet, 3-Class Trimap, and Pretrained Classifiers on identical inputs.
3. **Automated Benchmark on External Images:** Built-in validation across unseen samples in [`test images/`](test%20images/):
   - `Abyssinian_10.jpg` (Cat)
   - `american_bulldog_131.jpg` (Dog)
   - `american_pit_bull_terrier_14.jpg` (Dog)
   - `basset_hound_148.jpg` (Dog)
   - `beagle_84.jpg` (Dog)

### Python Single-Image Evaluation Snippet:
```python
import torch
from CSE428_Model_Evaluator_Demo import evaluate_and_display, DeepMultiTaskUNet

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = DeepMultiTaskUNet(use_attention=True, seg_classes=1).to(device)
model.load_state_dict(torch.load('notebooks/final executed ones/raw_output/best_att_binary.pth', map_location=device))
model.eval()

# Evaluates segmentation mask, breed classification, and confidence
evaluate_and_display(model, 'test images/beagle_84.jpg', model_title='Attention U-Net Demo')
```

---

## 📁 Repository Structure

```
pet-image-analysis-classification-detection/
├── CSE428 Summer 26 Project Guidelines.pdf  # Official project rubric & requirements
├── LICENSE                                 # MIT Open-Source License
├── README.md                               # Master project documentation
├── requirements.txt                        # Python dependencies
├── setup_env.ps1                           # Automated setup script for CUDA & PyTorch
│
├── assets/                                 # Publication-grade figures & benchmark plots
│   ├── 3x3_dataset_overlays.png
│   ├── breed_distribution.png
│   ├── mask_preprocessing_before_after.png
│   ├── base_unet_training_curves.png
│   ├── attention_unet_training_curves.png
│   ├── confusion_matrix_annotated.png
│   ├── test_metrics_comparison_bars.png
│   ├── prediction_sample_1.png
│   ├── prediction_sample_2.png
│   ├── bonus1_classifier_comparison.png
│   ├── bonus2_augmentation_ablation.png
│   ├── bonus3_trimap_segmentation.png
│   ├── bonus4_hparam_search.png
│   └── bonus5_bifpn_prediction.png
│
├── docs/                                   # In-depth technical documentation
│   ├── PROJECT_GUIDELINES_MAPPING.md       # Exact rubric compliance matrix
│   ├── ARCHITECTURE.md                     # Mathematical derivations & network schematics
│   ├── BENCHMARK_RESULTS.md                # Per-breed classification reports & statistics
│   └── EVALUATION_GUIDE.md                 # Live demo walkthrough & viva defense notes
│
├── notebooks/final executed ones/          # Fully executed Jupyter notebooks
│   ├── 428_raw.ipynb                       # Core tasks: Base U-Net & Attention U-Net
│   ├── bonus123.ipynb                      # Bonuses 1, 2, and 3: Classifiers, Augmentation, Trimap
│   ├── bonus-4.ipynb                       # Bonus 4: 9-trial Hyperparameter Grid Search
│   ├── bonus5.ipynb                        # Bonus 5: EfficientDet BiFPN Decoder
│   ├── CSE428_Model_Evaluator_Demo.ipynb   # Interactive Universal Evaluator & Demo Suite
│   │
│   ├── raw_output/                         # Core checkpoints, confusion matrix, reports
│   │   ├── best_att_binary.pth             # Trained Attention U-Net weights (13.3 MB)
│   │   ├── best_base_binary.pth            # Trained Base U-Net weights (13.2 MB)
│   │   ├── confusion_matrix_annotated.png
│   │   └── per_breed_classification_report.csv
│   │
│   ├── bonus123output/                     # Bonus 1, 2, 3 checkpoints
│   │   ├── best_att_3.pth                  # 3-Class Attention U-Net weights (7.4 MB)
│   │   ├── best_base_3.pth                 # 3-Class Base U-Net weights (7.3 MB)
│   │   ├── bonus1_DenseNet121_pretrained.pth
│   │   ├── bonus1_EfficientNetB0_pretrained.pth
│   │   └── bonus1_MobileNetV2_pretrained.pth
│   │
│   ├── bonus4 output/                      # Bonus 4 results & indices
│   └── bonus5 output/results/              # Bonus 5 BiFPN checkpoints
│       └── bonus5_bifpn_unet.pth           # Trained BiFPN-UNet weights (4.8 MB)
│
└── test images/                            # External unseen test images for evaluation
    ├── Abyssinian_10.jpg
    ├── american_bulldog_131.jpg
    ├── american_pit_bull_terrier_14.jpg
    ├── basset_hound_148.jpg
    └── beagle_84.jpg
```

---

## 🚀 Installation & Quickstart

### 1. Clone the Repository
```bash
git clone https://github.com/WalidMahmood/pet-image-analysis-classification-detection.git
cd pet-image-analysis-classification-detection
```

### 2. Environment Setup
Create a virtual environment and install dependencies:
```bash
# Optional: create virtual environment
python -m venv venv
# Windows:
.\venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate

# Install PyTorch with CUDA 12.1 (recommended for NVIDIA GPU):
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Install remaining dependencies:
pip install -r requirements.txt
```
*(On Windows, you can also run `.\setup_env.ps1`)*

### 3. Dataset Download (Oxford-IIIT Pet)
If training from scratch or running data exploration:
```bash
# Download official Oxford-IIIT Pet dataset into an 'archive' folder:
# http://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz
# http://www.robots.ox.ac.uk/~vgg/data/pets/data/annotations.tar.gz
```
The demo notebook [`CSE428_Model_Evaluator_Demo.ipynb`](notebooks/final%20executed%20ones/CSE428_Model_Evaluator_Demo.ipynb) can run inference on [`test images/`](test%20images/) immediately without downloading the full 780 MB dataset.

---

## 📚 Academic References

1. **U-Net:** Ronneberger, O., Fischer, P., & Brox, T. (2015). *U-Net: Convolutional Networks for Biomedical Image Segmentation.* MICCAI 2015. [arXiv:1505.04597](https://arxiv.org/abs/1505.04597).
2. **Attention U-Net:** Oktay, O., Schlemper, J., Folgoc, L. L., et al. (2018). *Attention U-Net: Learning Where to Look for the Pancreas.* MIDL 2018. [arXiv:1804.03999](https://arxiv.org/abs/1804.03999).
3. **EfficientDet & BiFPN:** Tan, M., Pang, R., & Le, Q. V. (2020). *EfficientDet: Scalable and Efficient Object Detection.* CVPR 2020. [arXiv:1911.09070](https://arxiv.org/abs/1911.09070).
4. **Oxford-IIIT Pet Dataset:** Parkhi, O. M., Vedaldi, A., Zisserman, A., & Jawahar, C. V. (2012). *Cats and Dogs.* CVPR 2012. [Link](https://www.robots.ox.ac.uk/~vgg/data/pets/).
5. **Albumentations:** Buslaev, A., et al. (2020). *Albumentations: Fast and Flexible Image Augmentations.* Information, 11(2), 125.

---

## 👥 Authors & Contributors

* **Walid Mahmood** — GitHub: [@WalidMahmood](https://github.com/WalidMahmood) • Email: `walid.mahmood@g.bracu.ac.bd`
* **Sayma Shahid** — GitHub: [@Sayma34](https://github.com/Sayma34) • Email: `saymashahid34@gmail.com`

---

## 📄 License & Acknowledgments

* **Course:** CSE428 (Computer Vision & Pattern Recognition / Deep Learning)
* **Code License:** Released under the [MIT License](LICENSE).
* **Dataset License:** Oxford-IIIT Pet Dataset is provided by the Visual Geometry Group (VGG), University of Oxford for research and educational purposes.

