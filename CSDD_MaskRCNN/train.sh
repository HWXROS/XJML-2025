#!/bin/bash
# 多 GPU 分布式训练脚本
# 使用方法: bash train.sh [GPU数量] [配置文件]

GPUS=${1:-4}
CONFIG=${2:-"configs/mask_rcnn_r101_fpn_csdd_4gpu.py"}

echo "========================================="
echo "多 GPU 分布式训练"
echo "========================================="
echo "GPU 数量: $GPUS"
echo "配置文件: $CONFIG"
echo "========================================="

# 设置环境变量
export OMP_NUM_THREADS=1
export PYTHONUNBUFFERED=1

# 使用 torchrun 启动分布式训练
torchrun \
    --nproc_per_node=$GPUS \
    --master_port=29500 \
    src/step3_train_dist.py \
    --config $CONFIG \
    --launcher pytorch\

echo "========================================="
echo "训练完成!"
echo "========================================="