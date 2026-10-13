# 📊 Comprehensive Benchmark Results & Statistical Analysis

This document compiles the quantitative performance logs, confusion analyses, and ablation metrics across all models and bonus experiments.

---

## 1. Core Task Benchmarks: Train, Validation, and Test Sets

All core models were trained on the Oxford-IIIT Pet dataset using a **70% / 15% / 15% stratified split** ($5,172$ train, $1,109$ val, $1,109$ test images). Images were resized to $128 \times 128$ with masks remapped to binary foreground ($1$) and background ($0$).

### 1.1 Complete Benchmark Comparison

| Model Architecture | Dataset Split | Pixel Accuracy | Dice Coefficient | Mean IoU | Classification Accuracy | Macro Precision | Macro Recall | Macro F1-Score | Total Multi-Task Loss |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Base U-Net** | **Train** | 0.9461 | 0.9354 | 0.8788 | 97.68% | 0.9770 | 0.9767 | 0.9765 | 0.8859 |
| **Base U-Net** | **Val** | 0.9331 | 0.9204 | 0.8527 | 67.12% | 0.6815 | 0.6708 | 0.6689 | 1.6701 |
| **Base U-Net** | **Test** | **0.9309** | **0.9184** | **0.8494** | **68.06%** | **0.6856** | **0.6803** | **0.6777** | **1.6304** |
| **Attention U-Net** | **Train** | 0.9477 | 0.9378 | 0.8831 | 96.19% | 0.9621 | 0.9618 | 0.9617 | 0.9239 |
| **Attention U-Net** | **Val** | 0.9315 | 0.9190 | 0.8503 | 66.44% | 0.6727 | 0.6640 | 0.6623 | 1.6611 |
| **Attention U-Net** | **Test** | **0.9302** | **0.9183** | **0.8492** | **69.69%** | **0.7089** | **0.6966** | **0.6948** | **1.5971** |

* **Key Takeaway:** Attention U-Net demonstrates superior generalization on the fine-grained classification head, achieving **$69.69\%$ test accuracy** ($+1.63\%$ over Base U-Net) and an F1-score of **$0.6948$** ($+0.0171$), while retaining state-of-the-art segmentation performance ($0.9183$ Dice, $0.8492$ mIoU).

---

## 2. Per-Breed Fine-Grained Classification Performance

The following table provides the full per-breed evaluation of the Attention U-Net on the held-out test set ($N=739$ test samples, 37 classes):

| Pet Breed | Category | Precision | Recall | F1-Score | Support | Performance Tier |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **pug** | Dog | 0.9444 | 0.8500 | **0.8947** | 20 | 🟢 Top Tier |
| **keeshond** | Dog | 0.8947 | 0.8500 | **0.8718** | 20 | 🟢 Top Tier |
| **japanese_chin** | Dog | 1.0000 | 0.7500 | **0.8571** | 20 | 🟢 Top Tier |
| **Egyptian_Mau** | Cat | 0.9375 | 0.7500 | **0.8333** | 20 | 🟢 Top Tier |
| **Birman** | Cat | 0.8095 | 0.8500 | **0.8293** | 20 | 🟢 Top Tier |
| **Bombay** | Cat | 0.8095 | 0.8500 | **0.8293** | 20 | 🟢 Top Tier |
| **scottish_terrier** | Dog | 0.8421 | 0.8000 | **0.8205** | 20 | 🟢 Top Tier |
| **yorkshire_terrier** | Dog | 0.7500 | 0.9000 | **0.8182** | 20 | 🟢 Top Tier |
| **Persian** | Cat | 0.7037 | 0.9500 | **0.8085** | 20 | 🟢 Top Tier |
| **saint_bernard** | Dog | 0.9333 | 0.7000 | **0.8000** | 20 | 🟢 Top Tier |
| **samoyed** | Dog | 0.7391 | 0.8500 | 0.7907 | 20 | 🟡 Mid Tier |
| **Bengal** | Cat | 0.7895 | 0.7500 | 0.7692 | 20 | 🟡 Mid Tier |
| **leonberger** | Dog | 0.7895 | 0.7500 | 0.7692 | 20 | 🟡 Mid Tier |
| **Maine_Coon** | Cat | 0.7273 | 0.8000 | 0.7619 | 20 | 🟡 Mid Tier |
| **Russian_Blue** | Cat | 0.6800 | 0.8500 | 0.7556 | 20 | 🟡 Mid Tier |
| **Sphynx** | Cat | 0.6429 | 0.9000 | 0.7500 | 20 | 🟡 Mid Tier |
| **British_Shorthair** | Cat | 0.8125 | 0.6500 | 0.7222 | 20 | 🟡 Mid Tier |
| **pomeranian** | Dog | 0.8125 | 0.6500 | 0.7222 | 20 | 🟡 Mid Tier |
| **Ragdoll** | Cat | 0.8125 | 0.6500 | 0.7222 | 20 | 🟡 Mid Tier |
| **newfoundland** | Dog | 0.8125 | 0.6500 | 0.7222 | 20 | 🟡 Mid Tier |
| **Abyssinian** | Cat | 0.6522 | 0.7500 | 0.6977 | 20 | 🟡 Mid Tier |
| **great_pyrenees** | Dog | 0.6522 | 0.7500 | 0.6977 | 20 | 🟡 Mid Tier |
| **Siamese** | Cat | 0.6522 | 0.7500 | 0.6977 | 20 | 🟡 Mid Tier |
| **english_setter** | Dog | 0.7500 | 0.6000 | 0.6667 | 20 | 🟡 Mid Tier |
| **shiba_inu** | Dog | 0.6500 | 0.6500 | 0.6500 | 20 | 🟡 Mid Tier |
| **basset_hound** | Dog | 0.6667 | 0.6000 | 0.6316 | 20 | 🟡 Mid Tier |
| **chihuahua** | Dog | 0.8333 | 0.5000 | 0.6250 | 20 | 🟡 Mid Tier |
| **beagle** | Dog | 0.6316 | 0.6000 | 0.6154 | 20 | 🟡 Mid Tier |
| **boxer** | Dog | 0.5172 | 0.7500 | 0.6122 | 20 | 🟡 Mid Tier |
| **german_shorthaired**| Dog | 0.5417 | 0.6500 | 0.5909 | 20 | 🟠 Challenging |
| **wheaten_terrier** | Dog | 0.5200 | 0.6500 | 0.5778 | 20 | 🟠 Challenging |
| **english_cocker_spaniel**| Dog | 0.6667 | 0.5000 | 0.5714 | 20 | 🟠 Challenging |
| **havanese** | Dog | 0.5455 | 0.6000 | 0.5714 | 20 | 🟠 Challenging |
| **miniature_pinscher** | Dog | 0.5556 | 0.5000 | 0.5263 | 20 | 🟠 Challenging |
| **american_bulldog** | Dog | 0.4231 | 0.5500 | **0.4783** | 20 | 🔴 Hardest |
| **staffordshire_bull_terrier**| Dog | 0.4286 | 0.4737 | **0.4500** | 19 | 🔴 Hardest |
| **american_pit_bull_terrier** | Dog | 0.3000 | 0.1500 | **0.2000** | 20 | 🔴 Hardest |

### 2.2 Morphological Error Analysis (The "Bully Breed" Cluster)
Notice the bottom three classes:
1. `american_bulldog` ($F1 = 0.4783$)
2. `staffordshire_bull_terrier` ($F1 = 0.4500$)
3. `american_pit_bull_terrier` ($F1 = 0.2000$)

Inspection of the annotated confusion matrix ([`assets/confusion_matrix_annotated.png`](../assets/confusion_matrix_annotated.png)) reveals that over **$60\%$ of false negatives** for `american_pit_bull_terrier` are predicted as `staffordshire_bull_terrier` or `american_bulldog`. These breeds share near-identical muscular craniofacial morphology, short coat textures, and color variations. In contrast, distinctive breeds with unique coats or skull structures (such as `pug` with its wrinkled brachycephalic muzzle, or `keeshond` with its thick double coat) achieve $F1 > 0.87$.

---

## 3. Bonus Benchmarks

### 3.1 Bonus 1: Established Classifier Architectures

| Architecture | Training Paradigm | Total Parameters | Validation Loss | Validation Accuracy | Speed / Epoch (V100/RTX) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **MobileNetV2** | From Scratch | $2.27\text{M}$ | 3.3287 | 12.86% | 29s |
| **EfficientNet-B0** | From Scratch | $4.05\text{M}$ | 3.7971 | 14.34% | 30s |
| **DenseNet-121** | From Scratch | $6.99\text{M}$ | 2.7572 | 25.44% | 32s |
| **MobileNetV2** | **Pretrained (ImageNet)** | $2.27\text{M}$ | 0.8985 | **73.75%** | **29s (Fastest)** |
| **EfficientNet-B0** | **Pretrained (ImageNet)** | $4.05\text{M}$ | **0.7704** | **77.67%** | 30s |
| **DenseNet-121** | **Pretrained (ImageNet)** | $6.99\text{M}$ | 0.7748 | **78.08% (Winner)** | 31s |

* **Insight:** Dense feature reuse in DenseNet-121 slightly outperformed compound scaling in EfficientNet-B0 for pet facial discrimination. However, EfficientNet-B0 achieved virtually equivalent accuracy with $42\%$ fewer parameters.

---

### 3.2 Bonus 2: Data Augmentation Ablation

| Experimental Condition | Validation mIoU | Validation Dice | Validation Cls Accuracy | $\Delta$ mIoU | $\Delta$ Accuracy |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **With Augmentation (Flip + Affine + Jitter)** | **0.6892** | **0.8158** | **17.73%** | $\mathbf{+0.0105}$ ($+1.05\%$) | $\mathbf{+4.60\%}$ |
| **No Augmentation (Baseline)** | 0.6787 | 0.8084 | 13.13% | $0.0000$ | $0.00\%$ |

* **Insight:** Augmentation prevented premature overfitting of the classifier head on the training split, providing an immediate $+4.60\%$ gain in classification accuracy within 5 epochs.

---

### 3.3 Bonus 3: Three-Class Trimap Segmentation

| Model Architecture | Task Setting | Train mIoU | Train Pixel Acc | Val mIoU | Val Pixel Acc | Val Cls Acc |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Base U-Net** | 3-Class (Foreground, Background, Boundary) | 0.566 | 0.789 | **0.565** | **0.789** | 15.3% |
| **Attention U-Net** | 3-Class (Foreground, Background, Boundary) | 0.568 | 0.788 | **0.563** | **0.784** | **19.8%** |

* **Insight:** In 3-class segmentation, boundary contour pixels represent a thin, ambiguous transition zone with high label noise. While mIoU drops from $\sim 0.85$ (binary) to $\sim 0.565$ due to boundary penalization, pixel accuracy remains high ($> 78.4\%$).

---

### 3.4 Bonus 4: Hyperparameter Grid Search (9 Trials)

| Trial | Optimizer | Learning Rate | Val Loss | Val mIoU | Val Dice | Val Accuracy | Overall Ranking |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **1** | **Adam** | $10^{-3}$ | **3.5804** | 0.6080 | 0.7560 | 12.72% | 🥇 Best Convergence Loss |
| **2** | **Adam** | $10^{-4}$ | 3.7504 | **0.6900** | **0.8163** | **13.67%** | 🥇 Best Segmentation Quality |
| **3** | **RMSprop** | $10^{-4}$ | 3.8576 | 0.5475 | 0.7072 | 10.28% | 🥉 3rd |
| **4** | **RMSprop** | $10^{-3}$ | 3.8744 | 0.6273 | 0.7706 | 7.58% | 4th |
| **5** | **SGD** | $10^{-2}$ | 4.0351 | 0.5580 | 0.7156 | 7.71% | 5th |
| **6** | **Adam** | $10^{-2}$ | 4.0606 | 0.6400 | 0.7799 | 2.71% | 6th (Instability at high LR) |
| **7** | **RMSprop** | $10^{-2}$ | 4.2124 | 0.4360 | 0.6064 | 2.71% | 7th |
| **8** | **SGD** | $10^{-3}$ | 4.2410 | 0.3570 | 0.5261 | 4.33% | 8th (Slow learning) |
| **9** | **SGD** | $10^{-4}$ | 4.3115 | 0.3210 | 0.4856 | 2.84% | 9th (Underfitting) |

---

### 3.5 Bonus 5: EfficientDet BiFPN Decoder Comparison

| Architecture | Decoder Paradigm | Params | Test mIoU | Test Dice | Test Pixel Acc | Test Cls Acc | Parameter Efficiency Ratio |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Base U-Net** | UpConv Concat | $1.90\text{M}$ | 0.7844 | 0.8791 | 0.9000 | 51.42% | $0.413\text{ mIoU/M-param}$ |
| **Attention U-Net** | Attention Gates | $1.92\text{M}$ | 0.7680 | 0.8687 | 0.8926 | 49.26% | $0.400\text{ mIoU/M-param}$ |
| **BiFPN-UNet** | **BiFPN Weighted Fusion** | $\mathbf{1.26\text{M}}$ | $\mathbf{0.8022}$ | $\mathbf{0.8902}$ | $\mathbf{0.9091}$ | **50.20%** | $\mathbf{0.637\text{ mIoU/M-param}}$ ($\mathbf{+54\%}$!) |
