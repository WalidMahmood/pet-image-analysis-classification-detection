# 🏗️ Neural Network Architecture Specifications

This document outlines the architectural blueprints, mathematical formulations, and dataflow pathways of the multi-task vision systems implemented in this repository.

---

## 1. Architectural Philosophy: Multi-Task Learning (MTL)

In real-world veterinary and domestic pet analysis, detecting the physical silhouette of an animal (**semantic segmentation**) and identifying its genetic lineage (**fine-grained breed classification**) are intrinsically coupled visual tasks.

```
                         [ Input Image: 3 × 128 × 128 ]
                                       │
                         ┌─────────────┴─────────────┐
                         │      U-Net Encoder        │
                         │   Feature Extraction      │
                         └─────────────┬─────────────┘
                                       │
                         [ Bottleneck Latent Space ]
                                       │
                ┌──────────────────────┴──────────────────────┐
                ▼                                             ▼
     [ Decoder / Skip Fusions ]                   [ Adaptive Global Pool ]
                │                                             │
      [ 1×1 Conv Head ]                             [ MLP Classifier Head ]
                │                                             │
                ▼                                             ▼
  [ Segmentation Mask: 1×128×128 ]              [ Breed Logits: 37 Classes ]
```

By enforcing a shared encoder backbone, the network learns representations that balance:
* **Spatial fidelity** (high-frequency edge and texture information needed for pixel segmentation).
* **Semantic abstraction** (invariance to pose, scale, and lighting needed for 37-breed classification).

---

## 2. Multi-Task Base U-Net (Ronneberger et al., 2015)

<div align="center">
  <img src="../assets/arch_base_unet.png" alt="Multi-Task Base U-Net Architecture" width="95%"/>
  <p><i>Figure 1: Architectural diagram of the Multi-Task Base U-Net with bottleneck classification head.</i></p>
</div>

### 2.1 Encoder Pathway
The encoder consists of four hierarchical contraction stages. Each stage employs a `DoubleConv` block:
$$\text{DoubleConv}(x) = \text{ReLU}\left(\text{BN}\left(\text{Conv2d}_{3 \times 3}\left(\text{ReLU}\left(\text{BN}\left(\text{Conv2d}_{3 \times 3}(x)\right)\right)\right)\right)\right)$$

Contraction is performed using $2 \times 2$ Max-Pooling with stride 2:
* **Level 1:** Input $3 \times 128 \times 128 \to$ Features $64 \times 128 \times 128 \to$ Pool $64 \times 64 \times 64$
* **Level 2:** Features $128 \times 64 \times 64 \to$ Pool $128 \times 32 \times 32$
* **Level 3:** Features $256 \times 32 \times 32 \to$ Pool $256 \times 16 \times 16$
* **Bottleneck:** Features $512 \times 8 \times 8$

### 2.2 Dual Task Heads
1. **Classification Head (Bottleneck):**
   $$z = \text{AdaptiveAvgPool2d}_{1 \times 1}(F_{\text{bottleneck}}) \in \mathbb{R}^{512}$$
   $$\hat{y}_{\text{cls}} = W_2 \cdot \text{Dropout}_{0.5}\left(\text{ReLU}\left(W_1 z + b_1\right)\right) + b_2, \quad \hat{y}_{\text{cls}} \in \mathbb{R}^{37}$$
2. **Segmentation Decoder:**
   Upsampling is performed via bilinear interpolation (or transposed convolutions) followed by channel concatenation with encoder skip connections:
   $$\tilde{F}_{l} = \text{Concat}\left(\text{Upsample}(F_{l+1}), F_{\text{encoder}, l}\right)$$
   $$\hat{Y}_{\text{seg}} = \text{Conv2d}_{1 \times 1}\left(\text{DecoderOutput}\right) \in \mathbb{R}^{1 \times 128 \times 128}$$

---

## 3. Multi-Task Attention U-Net (Oktay et al., 2018)

In vanilla U-Net, skip connections propagate raw encoder feature maps directly to the decoder. This introduces irrelevant background features (rugs, sofas, park grass) that can pollute segmentation boundaries.

The **Attention U-Net** incorporates an Additive Attention Gate (AG) at every skip interface:

<div align="center">
  <img src="../assets/arch_attention_unet.png" alt="Multi-Task Attention U-Net Architecture" width="98%"/>
  <p><i>Figure 2: Multi-Task Attention U-Net architecture with detailed zoom-in on the Additive Attention Gate module.</i></p>
</div>

### 3.1 Mathematical Formulation
Let $x_l \in \mathbb{R}^{C_l \times H \times W}$ denote the encoder skip feature map, and $g \in \mathbb{R}^{C_g \times H' \times W'}$ denote the coarser gating signal from the decoder.
1. The signals are linearly projected to intermediate dimension $F_{\text{int}} = C_l / 2$:
   $$q_{\text{att}} = \text{ReLU}\left(W_x^T x_l + W_g^T g + b_g\right)$$
2. The attention coefficient map $\alpha \in [0, 1]$ is generated:
   $$\alpha = \sigma\left(\psi^T q_{\text{att}} + b_\psi\right)$$
3. The skip features are modulated element-wise:
   $$\hat{x}_l = \alpha \odot x_l$$

This focuses spatial attention strictly on pet boundaries and contours, reducing false positives on cluttered backgrounds.

---

## 4. Bonus 5: EfficientDet BiFPN-UNet (Tan et al., 2020)

### 4.1 Theoretical Foundation
Standard Feature Pyramid Networks (FPN) and U-Net decoders rely solely on top-down or direct horizontal skip connections. However, different pyramid levels contribute unequally to the final segmentation decision.

The **BiFPN (Bidirectional Feature Pyramid Network)** introduces two key enhancements:
1. **Bidirectional Cross-Scale Connections:** Features flow both top-down and bottom-up repeatedly.
2. **Fast Normalized Feature Fusion:** Instead of unweighted addition or concatenation, each input feature is assigned a learnable, non-negative scalar weight:
   $$O = \sum_i \frac{w_i}{\epsilon + \sum_j w_j} \cdot I_i$$
   where $w_i \ge 0$ is enforced via a $\text{ReLU}$ activation, preventing numerical instability with $\epsilon = 10^{-4}$.

<div align="center">
  <img src="../assets/arch_bifpn_unet.png" alt="Bonus 5 BiFPN-UNet Architecture" width="95%"/>
  <p><i>Figure 3: EfficientDet BiFPN-UNet architecture featuring multi-scale bidirectional fusions.</i></p>
</div>

<div align="center">
  <img src="../assets/arch_comparison_overview.png" alt="Architectural Comparison Overview" width="95%"/>
  <p><i>Figure 4: Side-by-side comparison of Direct Concatenation, Additive Attention Gate, and BiFPN Weighted Fusion.</i></p>
</div>


```
  P4 (Bottleneck) ───────────────────────────► (w1·P4 + w2·P3_td) ──► P4_out
                          │                             ▲
                          ▼                             │
  P3 (Mid Level)  ───► (w1·P3 + w2·P4_up) ───► (w1·P3 + w2·P3_td + w3·P2_up) ──► P3_out
                          │                             ▲
                          ▼                             │
  P2 (Fine Level) ───► (w1·P2 + w2·P3_up) ─────────────────────────────────────► P2_out
```

### 4.2 Parameter & Computational Efficiency
* **Base U-Net Parameters:** $1.90\text{M}$ (or $3.44\text{M}$ in deep configuration)
* **Attention U-Net Parameters:** $1.92\text{M}$ (or $3.47\text{M}$)
* **BiFPN-UNet Parameters:** **$1.26\text{M}$** ($-34\%$ to $-63\%$ parameter reduction!)

Despite having significantly fewer parameters, BiFPN-UNet achieved the **highest test mIoU (0.8022)** among all models trained under identical controlled protocols.

---

## 5. Loss Formulation & Multi-Task Objective

The total optimization objective $\mathcal{L}_{\text{total}}$ is defined as:
$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{seg}} + \lambda \mathcal{L}_{\text{cls}}$$

### 5.1 Segmentation Loss ($\mathcal{L}_{\text{seg}}$)
To handle class imbalance (the pet foreground occupies $\sim 30\text{--}40\%$ of the image area), we combine Binary Cross-Entropy with Soft Dice Loss:
$$\mathcal{L}_{\text{seg}} = \mathcal{L}_{\text{BCE}}(\hat{Y}, Y) + \mathcal{L}_{\text{Dice}}(\hat{Y}, Y)$$
$$\mathcal{L}_{\text{Dice}} = 1 - \frac{2 \sum_{i=1}^N \hat{p}_i y_i + \epsilon}{\sum_{i=1}^N \hat{p}_i + \sum_{i=1}^N y_i + \epsilon}$$
where $\hat{p}_i = \sigma(\hat{y}_i)$ is the predicted probability and $\epsilon = 1.0$ is the Laplace smoothing parameter.

### 5.2 Classification Loss ($\mathcal{L}_{\text{cls}}$)
Standard multi-class categorical cross-entropy over 37 pet breeds:
$$\mathcal{L}_{\text{cls}} = -\sum_{c=1}^{37} y_c \log \frac{e^{\hat{z}_c}}{\sum_{k=1}^{37} e^{\hat{z}_k}}$$

Throughout all baseline experiments, $\lambda = 1.0$ was established as the balanced task weight.
