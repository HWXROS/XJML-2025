# CSDD 缺陷检测与分割项目

基于 **Mask R-CNN + ResNet-101-FPN** 的工业铸造件表面缺陷检测与分割系统。

## 项目概述

本项目使用 MMDetection 框架，在 CSDD 数据集上实现：
- **缺陷检测**：定位并识别表面缺陷，输出边界框和类别
- **缺陷分割**：像素级精确分割缺陷区域

### 数据集

CSDD (Casting Surface Defect Detection) 数据集：
- 图像数量：2100张高分辨率RGB图像
- 缺陷实例：56,356个专家标注实例
- 缺陷类别：Scratches（划痕）、Spots（斑点）
- 数据划分：训练集64%、验证集16%、测试集20%

### 评估指标

- **检测**：mAP、AP@0.5、AP@0.75、各类别AP
- **分割**：mIoU、各类别IoU

## 环境要求

- Python 3.8+
- PyTorch 2.0+ （需自行安装）
- CUDA 11.8+（需自行安装）
- GPU: RTX 4090 (24GB) 或同等显存

## 快速开始

### 1. 安装依赖（假设已有PyTorch环境）

```bash
bash setup.sh
```

或手动安装：
```bash
pip install -U openmim
mim install mmengine mmcv mmdet
pip install pycocotools opencv-python matplotlib seaborn tqdm tabulate scipy Pillow PyYAML
```

### 2. 准备数据

将CSDD数据集放入 `data/CSDD_raw/` 目录：

```
data/CSDD_raw/
├── CSDD_det/
│   └── CSDD_det/
│       ├── images/
│       │   ├── train2017/
│       │   ├── val2017/
│       │   └── test2017/
│       └── labels/
│           ├── train2017/
│           ├── val2017/
│           └── test2017/
└── CSDD_seg/
    └── CSDD_seg/
        ├── img/
        │   ├── train/
        │   ├── val/
        │   └── test/
        └── ground_truth/
            ├── train/
            ├── val/
            └── test/
```

### 3. 一键运行

```bash
bash run_all.sh
```

或分步执行：

```bash
# 步骤1：分析数据集
python src/step1_analyze_dataset.py

# 步骤2：转换数据格式
python src/step2_convert_to_coco.py

# 步骤3：训练模型
python src/step3_train.py --config configs/mask_rcnn_r101_fpn_csdd.py

# 步骤4：评估模型
python src/step4_evaluate.py \
    --config configs/mask_rcnn_r101_fpn_csdd.py \
    --checkpoint outputs/work_dirs/mask_rcnn_r101_fpn_csdd/best_coco_bbox_mAP.pth

# 步骤5：可视化结果
python src/step5_visualize.py \
    --config configs/mask_rcnn_r101_fpn_csdd.py \
    --checkpoint outputs/work_dirs/mask_rcnn_r101_fpn_csdd/best_coco_bbox_mAP.pth
```

## 项目结构

```
CSDD_MaskRCNN/
├── configs/
│   └── mask_rcnn_r101_fpn_csdd.py    # 模型配置
├── src/
│   ├── step1_analyze_dataset.py      # 数据分析
│   ├── step2_convert_to_coco.py      # 格式转换
│   ├── step3_train.py                # 模型训练
│   ├── step4_evaluate.py             # 模型评估
│   ├── step5_visualize.py            # 结果可视化
│   └── utils/                        # 工具函数
├── data/
│   ├── CSDD_raw/                     # 原始数据
│   └── CSDD_coco/                    # COCO格式数据
├── outputs/
│   ├── work_dirs/                    # 训练输出
│   ├── eval_results/                 # 评估结果
│   └── visualizations/               # 可视化图片
├── setup.sh                          # 环境安装
├── run_all.sh                        # 一键运行
└── requirements.txt                  # 依赖包
```

## 输出文件

训练完成后，主要输出文件：

| 文件 | 说明 |
|------|------|
| `outputs/eval_results/evaluation_report.md` | 完整评估报告 |
| `outputs/eval_results/evaluation_metrics.json` | 评估指标JSON |
| `outputs/visualizations/detection/` | 检测结果可视化 |
| `outputs/visualizations/segmentation/` | 分割结果可视化 |
| `outputs/visualizations/comparison/` | GT vs Pred对比 |
| `outputs/work_dirs/*/best_*.pth` | 最优模型权重 |

## 配置说明

主要配置项在 `configs/mask_rcnn_r101_fpn_csdd.py`：

```python
# 类别定义
CLASSES = ('Scratches', 'Spots')
NUM_CLASSES = 2

# 训练配置
train_dataloader = dict(batch_size=4)  # 根据显存调整
max_epochs = 36

# 输入尺寸
scale = (1024, 1024)

# 优化器
optimizer = dict(type='SGD', lr=0.02, momentum=0.9)
```

## 常见问题

### 1. 显存不足

减小batch_size：
```python
train_dataloader = dict(batch_size=2)  # 或1
```

### 2. 训练速度慢

- 检查GPU利用率
- 增加num_workers
- 使用混合精度训练：`python src/step3_train.py --amp`

### 3. 数据路径错误

修改 `src/step2_convert_to_coco.py` 中的路径配置：
```python
CSDD_DET_ROOT = "你的实际路径/CSDD_det/CSDD_det"
CSDD_SEG_ROOT = "你的实际路径/CSDD_seg/CSDD_seg"
```

## 参考

- [MMDetection](https://github.com/open-mmlab/mmdetection)
- [Mask R-CNN](https://arxiv.org/abs/1703.06870)
- [CSDD Dataset](https://github.com/...)
