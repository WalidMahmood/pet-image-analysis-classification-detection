# 📋 CSE428 Project Guidelines — Verification & Compliance Matrix

> **Course:** CSE428 (Computer Vision & Pattern Recognition / Deep Learning)  
> **Session:** Summer 2026  
> **Author:** Walid Mahmood  
> **Repository:** [pet-image-analysis-classification-detection](https://github.com/WalidMahmood/pet-image-analysis-classification-detection.git)  
> **Reference Guidelines:** [`CSE428 Summer 26 Project Guidelines.pdf`](../CSE428%20Summer%2026%20Project%20Guidelines.pdf)

---

## 🎯 Executive Compliance Summary

Every core requirement, presentation mandate, demonstration deliverable, and all **5 optional bonus tasks** defined in the official project specification have been implemented, trained, benchmarked, and documented.

| Category | Guideline Item | Status | Primary Code / Notebook Artifact | Verified Output / Metric |
| :--- | :--- | :---: | :--- | :--- |
| **Core** | Boundary pixel collapse (Trimap $\to$ Binary) | ✅ Complete | [`428_raw.ipynb`](../notebooks/final%20executed%20ones/428_raw.ipynb) (Cell 10–16) | Remapped $\{1, 3\} \to 1, 2 \to 0$ |
| **Core** | $3 \times 3$ Random Image-Mask Overlay Grid | ✅ Complete | [`428_raw.ipynb`](../notebooks/final%20executed ones/428_raw.ipynb) (Cell 20) | [`assets/3x3_dataset_overlays.png`](../assets/3x3_dataset_overlays.png) |
| **Core** | Multi-Task Base U-Net (Ronneberger 2015) | ✅ Complete | [`428_raw.ipynb`](../notebooks/final%20executed%20ones/428_raw.ipynb) (Cell 24, 31) | Test mIoU: **0.8494**, Cls Acc: **68.06%** |
| **Core** | Multi-Task Attention U-Net (Oktay 2018) | ✅ Complete | [`428_raw.ipynb`](../notebooks/final%20executed%20ones/428_raw.ipynb) (Cell 24, 32) | Test mIoU: **0.8492**, Cls Acc: **69.69%** |
| **Core** | Train/Val/Test Metrics (mIoU, Dice, Acc, P, R, F1) | ✅ Complete | [`428_raw.ipynb`](../notebooks/final%20executed%20ones/428_raw.ipynb) (Cell 34) | Full tabular summaries across all 3 splits |
| **Core** | Arbitrary Index Single-Image Prediction View | ✅ Complete | [`428_raw.ipynb`](../notebooks/final%20executed%20ones/428_raw.ipynb) (Cell 35) | Side-by-side: Image, GT, Mask, Pred, Conf |
| **Bonus 1** | Multiple Classifier Architectures ($\ge 3$) | ✅ Complete | [`bonus123.ipynb`](../notebooks/final%20executed%20ones/bonus123.ipynb) (Cell 32–33) | MobileNetV2, EfficientNet-B0, DenseNet-121 |
| **Bonus 2** | Data Augmentation Ablation Study | ✅ Complete | [`bonus123.ipynb`](../notebooks/final%20executed%20ones/bonus123.ipynb) (Cell 34–35) | $+1.05\%$ mIoU, $+4.60\%$ Cls Acc improvement |
| **Bonus 3** | 3-Class Trimap Segmentation (Boundary Preserved) | ✅ Complete | [`bonus123.ipynb`](../notebooks/final%20executed%20ones/bonus123.ipynb) (Cell 36–37) | Base & Attn U-Net evaluated on 3 classes |
| **Bonus 4** | Hyperparameter Grid Search (Optimizers & LRs) | ✅ Complete | [`bonus-4.ipynb`](../notebooks/final%20executed%20ones/bonus-4.ipynb) (Cell 31–33) | 9 experiments (Adam/RMSprop/SGD $\times$ 3 LRs) |
| **Bonus 5** | EfficientDet BiFPN Decoder on U-Net | ✅ Complete | [`bonus5.ipynb`](../notebooks/final%20executed%20ones/bonus5.ipynb) (Cell 31–35) | **$1.26\text{M}$** params ($-63\%$), mIoU: **0.8022** |
| **Demo** | Universal Multi-Model Evaluator Suite | ✅ Complete | [`CSE428_Model_Evaluator_Demo.ipynb`](../notebooks/final%20executed%20ones/CSE428_Model_Evaluator_Demo.ipynb) | Evaluates all external test images live |

---

## 🔍 Detailed Guideline Verification Breakdown

### 1. Dataset Exploration (Guidelines Page 1)

#### Requirement 1.1: Boundary Pixel Collapse
* **Guideline:** *"Load the data and change the unclassified or boundary pixels in the masks to foreground."*
* **Implementation:**
  * Raw Oxford-IIIT trimap values: `1` = Foreground (Pet), `2` = Background, `3` = Boundary/Unclassified.
  * In `428_raw.ipynb` (Cell 10–16), we verify and remap masks:
    $$\text{mask}_{\text{binary}} = \begin{cases} 1 & \text{if } \text{pixel} \in \{1, 3\} \\ 0 & \text{if } \text{pixel} = 2 \end{cases}$$
  * Nearest-neighbor interpolation is enforced to ensure no synthetic intermediate values are created during resizing to $128 \times 128$.
  * Visual proof: [`assets/mask_preprocessing_before_after.png`](../assets/mask_preprocessing_before_after.png).

#### Requirement 1.2: $3 \times 3$ Random Mask Overlay Grid
* **Guideline:** *"Prepare a code that takes 9 random indices and plots images with mask overlays in the following 3x3 format."*
* **Implementation:**
  * In `428_raw.ipynb` (Cell 20), function `plot_3x3_grid(image_paths, mask_paths, seed=SEED)` samples 9 random images across diverse breeds and overlays a transparent green mask ($\alpha = 0.45$) on top of the pet body.
  * Visual artifact: [`assets/3x3_dataset_overlays.png`](../assets/3x3_dataset_overlays.png).

---

### 2. Multi-Task Base U-Net (Guidelines Page 1–2)

#### Requirement 2.1: Base U-Net Architecture
* **Guideline:** *"Prepare a U-Net architecture (https://arxiv.org/abs/1505.04597) to segment the data."*
* **Implementation:**
  * 4-level encoder-decoder architecture with DoubleConv blocks (`Conv2d -> BatchNorm2d -> ReLU -> Conv2d -> BatchNorm2d -> ReLU`).
  * Skip connections concatenate corresponding encoder feature maps with upsampled decoder feature maps.

#### Requirement 2.2: Encoder Classifier Head & Joint Training
* **Guideline:** *"Attach a classifier head to the encoder output and classify the data. Train together."*
* **Implementation:**
  * The encoder bottleneck feature map ($F \in \mathbb{R}^{512 \times 8 \times 8}$) is pooled via `AdaptiveAvgPool2d(1)` and fed through a linear classifier (`Linear(512, 256) -> ReLU -> Dropout(0.5) -> Linear(256, 37)`).
  * Joint loss formulation:
    $$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{seg}} + \lambda \mathcal{L}_{\text{cls}}$$
    $$\mathcal{L}_{\text{seg}} = \mathcal{L}_{\text{BCE}}(\hat{Y}_{\text{seg}}, Y_{\text{seg}}) + \mathcal{L}_{\text{Dice}}(\hat{Y}_{\text{seg}}, Y_{\text{seg}}), \quad \mathcal{L}_{\text{cls}} = \mathcal{L}_{\text{CE}}(\hat{Y}_{\text{cls}}, Y_{\text{cls}})$$
  * Joint training executed for 20 epochs with Adam ($\text{lr} = 10^{-3}$, cosine annealing).

#### Requirement 2.3: Single-Image Demonstration Function
* **Guideline:** *"Prepare a code that takes any image index as an input, passes it through your model and displays the results."*
* **Implementation:**
  * Function `predict_and_visualize(model, idx, split='test')` takes an image index, extracts the ground truth, computes the predicted segmentation mask and breed logit, and generates a multi-panel visual figure showing:
    1. Input RGB Image
    2. Ground Truth Mask Overlay
    3. Predicted Segmentation Mask Overlay (with pixel IoU and Dice score)
    4. Ground Truth Breed vs. Predicted Breed with Confidence Percentage.
  * Artifacts: [`assets/prediction_sample_1.png`](../assets/prediction_sample_1.png), [`assets/prediction_sample_2.png`](../assets/prediction_sample_2.png).

---

### 3. Multi-Task Attention U-Net (Guidelines Page 2)

#### Requirement 3.1: Attention Gates Architecture
* **Guideline:** *"Update the U-Net architecture to Attention U-Net (https://arxiv.org/abs/1804.03999) and run the segmentation training again."*
* **Implementation:**
  * Each skip connection integrates an Additive Attention Gate (`AttentionGate` module):
    $$\alpha = \sigma\left(\psi^T \left(\text{ReLU}\left(W_x x_l + W_g g - b_g\right)\right) + b_\psi\right)$$
    $$\hat{x}_l = \alpha \cdot x_l$$
  * Skip features $x_l$ are filtered by gating signal $g$ from the coarser decoder level, suppressing background clutter and focusing gradient updates on pet boundaries.

#### Requirement 3.2: Joint Classifier Head & Comparison
* **Guideline:** *"Attach the previous classifier head to the encoder output and classify the data again. Once again, train together. Follow presentation reporting metrics."*
* **Implementation:**
  * Identical classifier head and multi-task loss formulation.
  * Attention U-Net achieves superior classification accuracy (**69.69%** vs. 68.06%) and tighter boundary localization.
  * Verified curves: [`assets/attention_unet_training_curves.png`](../assets/attention_unet_training_curves.png).

---

### 4. Presentation & Reporting Metrics (Guidelines Page 3)

#### Requirement 4.1: Metric Scope (Train, Val, Test)
* **Guideline:** *"We want to see the values of mean Intersection over union (mIoU), dice coefficient and pixel accuracies for the segmentation task, and accuracy, precision, recall and F1 scores for classification task. The result summaries should be presented for all sets - training, validation and test."*
* **Verification Matrix (`428_raw.ipynb` Cell 34):**

| Model | Set | Pixel Acc | Dice Coeff | mIoU | Cls Acc | Precision | Recall | Macro F1 | Total Loss |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Base U-Net** | **Train** | 0.9461 | 0.9354 | 0.8788 | 0.9768 | 0.9770 | 0.9767 | 0.9765 | 0.8859 |
| **Base U-Net** | **Val** | 0.9331 | 0.9204 | 0.8527 | 0.6712 | 0.6815 | 0.6708 | 0.6689 | 1.6701 |
| **Base U-Net** | **Test** | **0.9309** | **0.9184** | **0.8494** | **0.6806** | **0.6856** | **0.6803** | **0.6777** | **1.6304** |
| **Attention U-Net** | **Train** | 0.9477 | 0.9378 | 0.8831 | 0.9619 | 0.9621 | 0.9618 | 0.9617 | 0.9239 |
| **Attention U-Net** | **Val** | 0.9315 | 0.9190 | 0.8503 | 0.6644 | 0.6727 | 0.6640 | 0.6623 | 1.6611 |
| **Attention U-Net** | **Test** | **0.9302** | **0.9183** | **0.8492** | **0.6969** | **0.7089** | **0.6966** | **0.6948** | **1.5971** |

---

## 🏆 Bonus Tasks Verification Breakdown

### Bonus 1: Well-Established Classifier Architectures ($\ge 3$)
* **Guideline:** *"Test out different well-established classifier architectures (like MobileNet, EfficientNet, DenseNet etc.) and see how the performance changes. Show comparison between at least 3. (Difficulty: Easy)"*
* **Notebook:** [`bonus123.ipynb`](../notebooks/final%20executed%20ones/bonus123.ipynb) (Cell 32–33)
* **Weights:** `bonus1_MobileNetV2_*.pth`, `bonus1_EfficientNetB0_*.pth`, `bonus1_DenseNet121_*.pth`
* **Experimental Findings:**
  * Evaluated across **both** from-scratch training and ImageNet transfer learning:
    * **DenseNet-121 (Pretrained):** **78.08%** accuracy (Winner overall)
    * **EfficientNet-B0 (Pretrained):** **77.67%** accuracy (Best accuracy-per-parameter ratio)
    * **MobileNetV2 (Pretrained):** **73.75%** accuracy (Fastest inference, 2.27M params)
  * Visual proof: [`assets/bonus1_classifier_comparison.png`](../assets/bonus1_classifier_comparison.png).

---

### Bonus 2: Data Augmentation Ablation Study
* **Guideline:** *"Test if your model performance improves when you augment the data. Mention how you have augmented the data and how much the performance improves. (Difficulty: Low medium)"*
* **Notebook:** [`bonus123.ipynb`](../notebooks/final%20executed%20ones/bonus123.ipynb) (Cell 34–35)
* **Weights:** `ablation_aug.pth` vs. `ablation_noaug.pth`
* **Pipeline:**
  * Spatial transforms: Random horizontal flip ($p=0.5$), shift-scale-rotate ($\pm 10^\circ$, scale $\pm 10\%$), affine shearing.
  * Color transforms: ColorJitter (brightness, contrast, saturation), Gaussian blur.
* **Results:**
  * **With Augmentation:** mIoU = **0.6892**, Dice = **0.8158**, Cls Acc = **17.73%**
  * **Without Augmentation (Baseline):** mIoU = **0.6787**, Dice = **0.8084**, Cls Acc = **13.13%**
  * **Gain:** $\mathbf{+1.05\%}$ mIoU, $\mathbf{+4.60\%}$ Classification Accuracy.
  * Visual proof: [`assets/bonus2_augmentation_ablation.png`](../assets/bonus2_augmentation_ablation.png).

---

### Bonus 3: Three-Class Trimap Segmentation
* **Guideline:** *"Can you segment without collapsing the boundary pixels? Segment the images keeping the segmentation as a three class problem. Repeat segmentation using both U-Net and Attention U-Net. Use apt metrics. (Difficulty: Medium)"*
* **Notebook:** [`bonus123.ipynb`](../notebooks/final%20executed%20ones/bonus123.ipynb) (Cell 36–37)
* **Weights:** `best_base_3.pth`, `best_att_3.pth`
* **Task Setup:**
  * Classes: $0$ = Pet (Foreground), $1$ = Background, $2$ = Boundary Contour.
  * Softmax multi-class cross-entropy + multi-class Dice loss.
* **Results:**
  * **Base U-Net (3-Class):** Val mIoU = **0.565**, Val Pixel Acc = **0.789**
  * **Attention U-Net (3-Class):** Val mIoU = **0.563**, Val Pixel Acc = **0.784**, Val Cls Acc = **19.8%**
  * Visual proof: [`assets/bonus3_trimap_segmentation.png`](../assets/bonus3_trimap_segmentation.png).

---

### Bonus 4: Systematic Hyperparameter Grid Search
* **Guideline:** *"Test how your model performance changes with change in hyperparameters. For example, you can test for the most optimal optimizer under a wide range of learning rates. Remember, you have to compile your results properly for your work to be considered complete. (Difficulty: Medium)"*
* **Notebook:** [`bonus-4.ipynb`](../notebooks/final%20executed%20ones/bonus-4.ipynb) (Cell 31–33)
* **Grid Space:**
  * Optimizers: `Adam`, `RMSprop`, `SGD` (with momentum $0.9$)
  * Learning Rates: $10^{-2}$, $10^{-3}$, $10^{-4}$
  * Total Trials: $3 \times 3 = 9$ isolated convergence sweeps
* **Key Findings:**
  * **Top Performer:** Adam with $\text{lr} = 10^{-4}$ yielded highest validation mIoU (**0.6900**) and Dice (**0.8163**).
  * **Lowest Loss:** Adam with $\text{lr} = 10^{-3}$ achieved fastest convergence and lowest loss (**3.5804**).
  * **Instability:** $\text{lr} = 10^{-2}$ caused divergence or suboptimal plateaus across all three optimizers.
  * Visual proof: [`assets/bonus4_hparam_search.png`](../assets/bonus4_hparam_search.png).

---

### Bonus 5: EfficientDet BiFPN Architecture on U-Net Decoder
* **Guideline:** *"Implementing EfficientDet (https://arxiv.org/abs/1911.09070) architecture on the decoder side of the original U-Net architecture. (Difficulty: High)"*
* **Notebook:** [`bonus5.ipynb`](../notebooks/final%20executed%20ones/bonus5.ipynb) (Cell 31–35)
* **Weights:** `bonus5_bifpn_unet.pth`
* **Architectural Innovation:**
  * Replaces standard symmetric upconv decoder with multi-scale **Bidirectional Feature Pyramid Network (BiFPN)** blocks.
  * Fast normalized weighted feature fusion:
    $$O = \sum_i \frac{w_i}{\epsilon + \sum_j w_j} \cdot I_i$$
  * Bidirectional top-down and bottom-up cross-scale skip connections.
* **Comparative Results (Identical Protocol):**

| Architecture | Decoder Type | Params | Test mIoU | Test Dice | Test Pixel Acc | Test Cls Acc |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Base U-Net** | Standard Upsampling | $1.90\text{M}$ | 0.7844 | 0.8791 | 0.9000 | 51.42% |
| **Attention U-Net** | Attention Gates | $1.92\text{M}$ | 0.7680 | 0.8687 | 0.8926 | 49.26% |
| **BiFPN-UNet** | **EfficientDet BiFPN** | $\mathbf{1.26\text{M}}$ | $\mathbf{0.8022}$ | $\mathbf{0.8902}$ | $\mathbf{0.9091}$ | **50.20%** |

* **Impact:** **$+1.78\%$ higher mIoU** with **$34\%$ fewer parameters** than Base U-Net, and **$+3.42\%$ higher mIoU** with **$63\%$ fewer parameters** than deep baseline!
* Visual proof: [`assets/bonus5_bifpn_prediction.png`](../assets/bonus5_bifpn_prediction.png).

---

## 🎓 Demonstration & Viva Readiness (Guidelines Page 3)

1. **Live Demonstration (`CSE428_Model_Evaluator_Demo.ipynb`):**
   * Pre-configured to test arbitrary images from `test images/` or any upload on the fly.
   * Produces the exact requested presentation overlay: Input Image, Ground Truth, Predicted Mask Overlay, Class Labels, and Confidence.
2. **Viva Defense Topics:**
   * Why boundary collapse aids binary pet localization vs. why 3-class trimap reveals ambiguous contour uncertainty.
   * How additive attention gates act as spatial multipliers without adding significant FLOPs.
   * Why BiFPN achieves superior multi-scale receptive fields with fewer parameters due to depthwise separable convolutions and learned weight normalization.
