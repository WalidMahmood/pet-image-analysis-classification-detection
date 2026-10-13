# 🧪 Interactive Evaluation & Live Demonstration Guide

This guide provides instructions for faculty, evaluators, and researchers to test any model in this repository, run inference on unseen test images, and verify all claims.

---

## 1. Quickstart: Universal Model Evaluator Demo

The primary demonstration interface is implemented in:
👉 [`notebooks/final executed ones/CSE428_Model_Evaluator_Demo.ipynb`](../notebooks/final%20executed%20ones/CSE428_Model_Evaluator_Demo.ipynb)

### 1.1 How to Launch the Demo
1. Open terminal and navigate to the project root:
   ```powershell
   cd f:\pet-image-analysis-classification-detection
   jupyter notebook "notebooks/final executed ones/CSE428_Model_Evaluator_Demo.ipynb"
   ```
2. In Jupyter, select **Cell $\to$ Run All** (or run top-to-bottom).
3. The notebook automatically detects available GPUs (CUDA / RTX / V100 or CPU fallback) and resolves model weights from the local repository.

---

## 2. External Test Benchmark Suite (`test images/`)

The repository includes five unseen test images spanning diverse dog and cat breeds:

| Filename | True Pet Breed | Species | Characteristics |
| :--- | :--- | :---: | :--- |
| `Abyssinian_10.jpg` | `Abyssinian` | Cat | Slender body, warm ticked tabby coat, pointed ears |
| `american_bulldog_131.jpg` | `american_bulldog` | Dog | Robust muscular build, wide jaw, white/brindle coat |
| `american_pit_bull_terrier_14.jpg` | `american_pit_bull_terrier` | Dog | Short sleek coat, muscular head, high alertness |
| `basset_hound_148.jpg` | `basset_hound` | Dog | Long drooping ears, heavy bone structure, short legs |
| `beagle_84.jpg` | `beagle` | Dog | Tricolor hound pattern, floppy ears, medium build |

---

## 3. Running Single-Image Inference on Custom Images

You can evaluate any arbitrary image in Python using the unified evaluation helper functions:

```python
import torch
from PIL import Image
from CSE428_Model_Evaluator_Demo import evaluate_and_display, DeepMultiTaskUNet

# 1. Select device and load model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = DeepMultiTaskUNet(use_attention=True, seg_classes=1).to(device)

# 2. Load trained checkpoint
weights_path = 'notebooks/final executed ones/raw_output/best_att_binary.pth'
model.load_state_dict(torch.load(weights_path, map_location=device))
model.eval()

# 3. Run and visualize
IMAGE_PATH = 'test images/beagle_84.jpg'
evaluate_and_display(
    model=model,
    image_path=IMAGE_PATH,
    model_title='Attention U-Net Evaluation',
    task_type='binary_multitask'
)
```

### Output Format
The display function renders a side-by-side figure:
1. **Input Image:** Raw RGB image with normalized aspect ratio.
2. **Ground Truth Mask:** Transparent overlay of ground truth silhouette.
3. **Predicted Mask:** Model-predicted segmentation boundary (with exact pixel IoU and Dice score).
4. **Classification Bar:** Ground truth breed vs. top-1 predicted breed with softmax confidence percentage and Top-5 prediction distribution.

---

## 4. Multi-Model Side-by-Side Comparison

Section 4 of [`CSE428_Model_Evaluator_Demo.ipynb`](../notebooks/final%20executed%20ones/CSE428_Model_Evaluator_Demo.ipynb) runs a head-to-head comparison on the same test image across all models:

```
[ Input: american_pit_bull_terrier_14.jpg ]
 ├── Model 1: Raw Attention U-Net (3.47M Params) ──► IoU: 0.887 | Pred: Pit Bull (84.2%)
 ├── Model 2: Raw Base U-Net (3.44M Params)      ──► IoU: 0.862 | Pred: Pit Bull (79.8%)
 ├── Model 3: Bonus 5: BiFPN-UNet (1.26M Params) ──► IoU: 0.894 | Pred: Pit Bull (82.1%)
 └── Model 4: Bonus 3: 3-Class Trimap Model      ──► 3-Class Mask: Foreground / Background / Boundary
```

---

## 5. Viva Defense Quick Reference

During the project demonstration, evaluators may ask about key implementation decisions. Here are direct references to the codebase:

### Q1: "How did you handle the boundary pixels during preprocessing?"
* **Answer:** *"In `428_raw.ipynb` (Cell 10–16), Oxford-IIIT trimap values $\{1, 3\}$ were remapped to 1 (foreground pet) and $2$ was remapped to 0 (background). Nearest-neighbor interpolation was strictly preserved to avoid synthetic float artifacts. In Bonus 3 (`bonus123.ipynb` Cell 36), we also trained the models on the uncollapsed 3-class trimap to observe how boundary uncertainty affects mIoU."*

### Q2: "How is the network trained jointly on two different tasks?"
* **Answer:** *"We use a shared encoder. The segmentation decoder reconstructs spatial masks from skip connections, while an MLP classification head sits on the bottleneck pooled latent vector ($F \in \mathbb{R}^{512}$). The total loss is $\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{BCE+Dice}} + \lambda \mathcal{L}_{\text{CE}}$, balancing pixel segmentation with breed discrimination."*

### Q3: "What advantage did BiFPN provide over Attention U-Net in Bonus 5?"
* **Answer:** *"BiFPN employs bidirectional multi-scale feature flows with learnable fast normalized fusion ($O = \sum \frac{w_i}{\sum w_j + \epsilon} I_i$). This achieved a test mIoU of 0.8022 while cutting model parameters from $1.90\text{M}$ down to $1.26\text{M}$ (a $34\%$ parameter reduction)."*
