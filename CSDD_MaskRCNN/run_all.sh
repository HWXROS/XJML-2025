#!/bin/bash
# CSDD Mask R-CNN 一键运行脚本
# 按顺序执行所有步骤

set -e  # 遇到错误立即退出

echo "========================================="
echo "CSDD Mask R-CNN 完整流程"
echo "========================================="

# 切换到项目根目录
cd "$(dirname "$0")"

# 检查环境
echo "[检查] 验证Python环境..."
python -c "import torch; import mmdet; print('环境OK')" || {
    echo "[ERROR] 环境未配置，请先运行: bash setup.sh"
    exit 1
}

# 第1步：分析数据集
echo ""
echo "========================================="
echo "[步骤 1/5] 分析数据集"
echo "========================================="
python src/step1_analyze_dataset.py

# 第2步：转换数据格式
echo ""
echo "========================================="
echo "[步骤 2/5] 转换数据格式"
echo "========================================="
python src/step2_convert_to_coco.py

# 第3步：训练模型
echo ""
echo "========================================="
echo "[步骤 3/5] 训练模型"
echo "========================================="
python src/step3_train.py --config configs/mask_rcnn_r101_fpn_csdd.py

# 找到最新的checkpoint
WORK_DIR="outputs/work_dirs/mask_rcnn_r101_fpn_csdd"
if [ -d "$WORK_DIR" ]; then
    # 尝试找best checkpoint，否则用最新的epoch
    CHECKPOINT=$(find "$WORK_DIR" -name "best_*.pth" -type f | head -1)
    if [ -z "$CHECKPOINT" ]; then
        CHECKPOINT=$(find "$WORK_DIR" -name "epoch_*.pth" -type f | sort -V | tail -1)
    fi
else
    echo "[ERROR] 训练输出目录不存在"
    exit 1
fi

if [ -z "$CHECKPOINT" ]; then
    echo "[ERROR] 找不到模型权重文件"
    exit 1
fi

echo "使用模型权重: $CHECKPOINT"

# 第4步：评估模型
echo ""
echo "========================================="
echo "[步骤 4/5] 评估模型"
echo "========================================="
python src/step4_evaluate.py \
    --config configs/mask_rcnn_r101_fpn_csdd.py \
    --checkpoint "$CHECKPOINT" \
    --output-dir outputs/eval_results

# 第5步：可视化结果
echo ""
echo "========================================="
echo "[步骤 5/5] 可视化结果"
echo "========================================="
python src/step5_visualize.py \
    --config configs/mask_rcnn_r101_fpn_csdd.py \
    --checkpoint "$CHECKPOINT" \
    --output-dir outputs/visualizations \
    --num-images 50

echo ""
echo "========================================="
echo "全部完成!"
echo "========================================="
echo ""
echo "输出文件:"
echo "  - 评估报告: outputs/eval_results/evaluation_report.md"
echo "  - 评估指标: outputs/eval_results/evaluation_metrics.json"
echo "  - 检测可视化: outputs/visualizations/detection/"
echo "  - 分割可视化: outputs/visualizations/segmentation/"
echo "  - 对比可视化: outputs/visualizations/comparison/"
echo "  - 模型权重: $CHECKPOINT"
echo ""
