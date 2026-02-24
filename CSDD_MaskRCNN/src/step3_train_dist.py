#!/usr/bin/env python3
"""
多 GPU 分布式训练脚本
使用 MMDetection 的分布式训练功能
"""

import os
import sys
import argparse
from datetime import datetime

# 设置多进程启动方式
import torch.multiprocessing as mp
try:
    mp.set_start_method('spawn', force=True)
except RuntimeError:
    pass


def parse_args():
    parser = argparse.ArgumentParser(description='多GPU训练Mask R-CNN')
    parser.add_argument(
        '--config',
        type=str,
        default='configs/mask_rcnn_r101_fpn_csdd_4gpu.py',
        help='配置文件路径'
    )
    parser.add_argument(
        '--work-dir',
        type=str,
        default=None,
        help='工作目录'
    )
    parser.add_argument(
        '--resume',
        type=str,
        default=None,
        help='从checkpoint恢复训练'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='随机种子'
    )
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='pytorch',
        help='分布式训练启动器'
    )
    parser.add_argument(
        '--local_rank',
        '--local-rank',
        type=int,
        default=0,
        help='本地进程rank'
    )
    return parser.parse_args()


def main():
    args = parse_args()
    
    # 检查配置文件
    if not os.path.exists(args.config):
        print(f"[ERROR] 配置文件不存在: {args.config}")
        return
    
    # 导入MMDetection
    try:
        from mmengine.config import Config
        from mmengine.runner import Runner
        from mmengine.dist import init_dist, get_rank
        import mmdet
    except ImportError as e:
        print(f"[ERROR] 请先安装MMDetection: {e}")
        return
    
    # 初始化分布式环境
    if args.launcher != 'none':
        init_dist(args.launcher)
    
    # 只在主进程打印信息
    rank = get_rank()
    if rank == 0:
        print("=" * 60)
        print("CSDD Mask R-CNN 多GPU训练")
        print("=" * 60)
        print(f"配置文件: {args.config}")
        print(f"启动器: {args.launcher}")
        print(f"随机种子: {args.seed}")
        print(f"MMDetection 版本: {mmdet.__version__}")
        print("=" * 60)
    
    # 加载配置
    cfg = Config.fromfile(args.config)
    
    # 设置分布式
    cfg.launcher = args.launcher
    
    # 设置工作目录
    if args.work_dir:
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        cfg.work_dir = f'outputs/work_dirs/mask_rcnn_r101_csdd_{timestamp}'
    
    os.makedirs(cfg.work_dir, exist_ok=True)
    
    if rank == 0:
        print(f"工作目录: {cfg.work_dir}")
    
    # 设置随机种子
    cfg.seed = args.seed
    
    # 设置resume
    if args.resume:
        cfg.resume = True
        cfg.load_from = args.resume
    
    # 构建Runner并训练
    runner = Runner.from_cfg(cfg)
    
    if rank == 0:
        print("\n开始训练...")
        print("-" * 60)
    
    runner.train()
    
    if rank == 0:
        print("\n" + "=" * 60)
        print("训练完成!")
        print("=" * 60)
        print(f"模型保存在: {cfg.work_dir}")


if __name__ == '__main__':
    main()
    os._exit(0)