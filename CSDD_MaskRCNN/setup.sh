#!/bin/bash
# CSDD Mask R-CNN 环境安装脚本
# 假设已安装 PyTorch 和 CUDA

echo "========================================="
echo "CSDD Mask R-CNN 环境安装"
echo "========================================="

# 检查PyTorch是否已安装
pip install "numpy<2.0"
echo "[1/4] 检查PyTorch..."
python -c "import torch; print(f'PyTorch版本: {torch.__version__}')" || {
    echo "[ERROR] PyTorch未安装，请先安装PyTorch"
    exit 1
}

python -c "import torch; print(f'CUDA可用: {torch.cuda.is_available()}')"

conda install -c conda-forge libgl -y
pip install "opencv-python==4.10.0.84"

# 安装MMDetection生态
echo "[2/4] 安装MMEngine, MMCV..."
pip install -U openmim
mim install mmengine
mim install "mmcv==2.1.0"

# 安装MMDetection
echo "[3/4] 安装MMDetection..."
mim install mmdet

# 安装其他依赖
echo "[4/4] 安装其他依赖..."
pip install pycocotools  matplotlib seaborn tqdm tabulate scipy Pillow PyYAML

echo "========================================="
echo "安装完成！"
echo "========================================="

# 验证安装
echo ""
echo "验证安装："
python3 -c "import torch; print(f'  PyTorch: {torch.__version__}')"
python3 -c "import torch; print(f'  CUDA: {torch.cuda.is_available()}')"
python3 -c "import cv2; print(f'  OpenCV: {cv2.__version__}')"
python3 -c "import mmcv; print(f'  MMCV: {mmcv.__version__}')"
python3 -c "import mmdet; print(f'  MMDetection: {mmdet.__version__}')"