# Bismillah - Setup for RTX 3050 8GB local + Kaggle
Write-Host "=== Creating venv (optional) ==="
# Option A: use existing Python 3.13 (Windows Store)
# python -m venv venv
# .\venv\Scripts\Activate.ps1

Write-Host "=== Installing PyTorch CUDA 12.1 (RTX 3050) ==="
pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt

Write-Host "=== Verifying ==="
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
python -c "import albumentations, sklearn, skimage; print('All deps OK')"
