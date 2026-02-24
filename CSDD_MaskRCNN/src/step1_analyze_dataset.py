#!/usr/bin/env python3
"""
第1步：数据集分析
分析CSDD数据集的基本统计信息

功能:
- 统计图像数量
- 统计各类别实例数量
- 分析目标尺寸分布
- 检查数据完整性
"""

import os
import sys
import cv2
import numpy as np
from collections import defaultdict
from tqdm import tqdm
import json

# 添加项目路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.mask_utils import load_mask, extract_instances_from_mask


# ======================== 配置 ========================
# 根据你的实际路径修改
CSDD_DET_ROOT = "data/CSDD_raw/CSDD_det/CSDD_det"
CSDD_SEG_ROOT = "data/CSDD_raw/CSDD_seg/CSDD_seg"

# 类别定义
CLASSES = {
    0: "Background",
    1: "Scratch",
    2: "Spot",
    3: "Rust"
}
NUM_CLASSES = 3  # 不含背景

# 数据集划分
SPLITS = {
    "train": {"det": "train2017", "seg": "train"},
    "val": {"det": "val2017", "seg": "val"},
    "test": {"det": "test2017", "seg": "test"}
}


def analyze_detection_labels(label_dir: str) -> dict:
    """
    分析YOLO格式的检测标注
    
    Args:
        label_dir: 标注文件目录
    
    Returns:
        统计信息字典
    """
    stats = {
        "total_labels": 0,
        "total_instances": 0,
        "class_counts": defaultdict(int),
        "bbox_widths": [],
        "bbox_heights": [],
        "bbox_areas": [],
        "instances_per_image": []
    }
    
    if not os.path.exists(label_dir):
        print(f"[WARNING] 标注目录不存在: {label_dir}")
        return stats
    
    label_files = [f for f in os.listdir(label_dir) if f.endswith('.txt')]
    stats["total_labels"] = len(label_files)
    
    for label_file in tqdm(label_files, desc="分析检测标注"):
        label_path = os.path.join(label_dir, label_file)
        instance_count = 0
        
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    class_id = int(parts[0])
                    x_center, y_center, width, height = [float(x) for x in parts[1:5]]
                    
                    stats["class_counts"][class_id] += 1
                    stats["bbox_widths"].append(width)
                    stats["bbox_heights"].append(height)
                    stats["bbox_areas"].append(width * height)
                    stats["total_instances"] += 1
                    instance_count += 1
        
        stats["instances_per_image"].append(instance_count)
    
    return stats


def analyze_segmentation_masks(mask_dir: str, num_classes: int = 2) -> dict:
    """
    分析分割掩膜
    
    Args:
        mask_dir: mask文件目录
        num_classes: 类别数量（不含背景）
    
    Returns:
        统计信息字典
    """
    stats = {
        "total_masks": 0,
        "total_instances": 0,
        "class_counts": defaultdict(int),
        "instance_areas": [],
        "instances_per_image": [],
        "mask_shapes": set()
    }
    
    if not os.path.exists(mask_dir):
        print(f"[WARNING] Mask目录不存在: {mask_dir}")
        return stats
    
    mask_files = [f for f in os.listdir(mask_dir) if f.endswith('_mask.png')]
    stats["total_masks"] = len(mask_files)
    
    for mask_file in tqdm(mask_files, desc="分析分割掩膜"):
        mask_path = os.path.join(mask_dir, mask_file)
        mask = load_mask(mask_path)
        
        stats["mask_shapes"].add(mask.shape)
        
        image_instances = 0
        for class_id in range(1, num_classes + 1):
            instances = extract_instances_from_mask(mask, class_id)
            stats["class_counts"][class_id] += len(instances)
            image_instances += len(instances)
            
            for inst_mask in instances:
                stats["instance_areas"].append(inst_mask.sum())
        
        stats["total_instances"] += image_instances
        stats["instances_per_image"].append(image_instances)
    
    # 转换set为list以便JSON序列化
    stats["mask_shapes"] = [str(s) for s in stats["mask_shapes"]]
    
    return stats


def check_data_consistency(det_root: str, seg_root: str, split: str) -> dict:
    """
    检查检测和分割数据的一致性
    
    Args:
        det_root: 检测数据根目录
        seg_root: 分割数据根目录
        split: 数据集划分 (train/val/test)
    
    Returns:
        一致性检查结果
    """
    det_split = SPLITS[split]["det"]
    seg_split = SPLITS[split]["seg"]
    
    det_images_dir = os.path.join(det_root, "images", det_split)
    det_labels_dir = os.path.join(det_root, "labels", det_split)
    seg_images_dir = os.path.join(seg_root, "img", seg_split)
    seg_masks_dir = os.path.join(seg_root, "ground_truth", seg_split)
    
    result = {
        "det_images": 0,
        "det_labels": 0,
        "seg_images": 0,
        "seg_masks": 0,
        "common_images": 0,
        "missing_labels": [],
        "missing_masks": []
    }
    
    # 检测数据
    if os.path.exists(det_images_dir):
        det_images = set(f.replace('.jpg', '') for f in os.listdir(det_images_dir) if f.endswith('.jpg'))
        result["det_images"] = len(det_images)
    else:
        det_images = set()
    
    if os.path.exists(det_labels_dir):
        det_labels = set(f.replace('.txt', '') for f in os.listdir(det_labels_dir) if f.endswith('.txt'))
        result["det_labels"] = len(det_labels)
    else:
        det_labels = set()
    
    # 分割数据
    if os.path.exists(seg_images_dir):
        seg_images = set(f.replace('.jpg', '') for f in os.listdir(seg_images_dir) if f.endswith('.jpg'))
        result["seg_images"] = len(seg_images)
    else:
        seg_images = set()
    
    if os.path.exists(seg_masks_dir):
        seg_masks = set(f.replace('_mask.png', '') for f in os.listdir(seg_masks_dir) if f.endswith('_mask.png'))
        result["seg_masks"] = len(seg_masks)
    else:
        seg_masks = set()
    
    # 检查一致性
    all_images = det_images | seg_images
    result["common_images"] = len(det_images & seg_images)
    
    # 找缺失的标注
    result["missing_labels"] = list(det_images - det_labels)[:10]  # 只显示前10个
    result["missing_masks"] = list(seg_images - seg_masks)[:10]
    
    return result


def main():
    """主函数"""
    print("=" * 60)
    print("CSDD 数据集分析")
    print("=" * 60)
    
    all_stats = {}
    
    for split in ["train", "val", "test"]:
        print(f"\n{'='*60}")
        print(f"分析 {split.upper()} 集")
        print("=" * 60)
        
        # 检测标注分析
        det_split = SPLITS[split]["det"]
        label_dir = os.path.join(CSDD_DET_ROOT, "labels", det_split)
        det_stats = analyze_detection_labels(label_dir)
        
        # 分割掩膜分析
        seg_split = SPLITS[split]["seg"]
        mask_dir = os.path.join(CSDD_SEG_ROOT, "ground_truth", seg_split)
        seg_stats = analyze_segmentation_masks(mask_dir, NUM_CLASSES)
        
        # 一致性检查
        consistency = check_data_consistency(CSDD_DET_ROOT, CSDD_SEG_ROOT, split)
        
        # 打印结果
        print(f"\n--- 检测标注统计 ---")
        print(f"标注文件数量: {det_stats['total_labels']}")
        print(f"实例总数: {det_stats['total_instances']}")
        print("各类别实例数:")
        for class_id, count in sorted(det_stats['class_counts'].items()):
            class_name = CLASSES.get(class_id + 1, f"Class {class_id}")  # YOLO类别从0开始
            print(f"  {class_name}: {count}")
        
        if det_stats['bbox_widths']:
            print(f"边界框宽度: 均值={np.mean(det_stats['bbox_widths']):.4f}, "
                  f"最小={min(det_stats['bbox_widths']):.4f}, "
                  f"最大={max(det_stats['bbox_widths']):.4f}")
            print(f"边界框高度: 均值={np.mean(det_stats['bbox_heights']):.4f}, "
                  f"最小={min(det_stats['bbox_heights']):.4f}, "
                  f"最大={max(det_stats['bbox_heights']):.4f}")
        
        print(f"\n--- 分割掩膜统计 ---")
        print(f"Mask文件数量: {seg_stats['total_masks']}")
        print(f"实例总数: {seg_stats['total_instances']}")
        print("各类别实例数:")
        for class_id, count in sorted(seg_stats['class_counts'].items()):
            class_name = CLASSES.get(class_id, f"Class {class_id}")
            print(f"  {class_name}: {count}")
        
        if seg_stats['mask_shapes']:
            print(f"Mask尺寸: {seg_stats['mask_shapes']}")
        
        print(f"\n--- 数据一致性检查 ---")
        print(f"检测图像: {consistency['det_images']}, 标注: {consistency['det_labels']}")
        print(f"分割图像: {consistency['seg_images']}, 掩膜: {consistency['seg_masks']}")
        print(f"共同图像: {consistency['common_images']}")
        
        if consistency['missing_labels']:
            print(f"缺失检测标注 (前10): {consistency['missing_labels']}")
        if consistency['missing_masks']:
            print(f"缺失分割掩膜 (前10): {consistency['missing_masks']}")
        
        # 保存统计信息
        all_stats[split] = {
            "detection": {k: v for k, v in det_stats.items() if k not in ['bbox_widths', 'bbox_heights', 'bbox_areas', 'instances_per_image']},
            "segmentation": {k: v for k, v in seg_stats.items() if k not in ['instance_areas', 'instances_per_image']},
            "consistency": consistency
        }
        all_stats[split]["detection"]["class_counts"] = dict(det_stats["class_counts"])
        all_stats[split]["segmentation"]["class_counts"] = dict(seg_stats["class_counts"])
    
    # 保存统计结果
    output_dir = "outputs/eval_results"
    os.makedirs(output_dir, exist_ok=True)
    stats_path = os.path.join(output_dir, "dataset_statistics.json")
    
    with open(stats_path, 'w', encoding='utf-8') as f:
        json.dump(all_stats, f, indent=2, ensure_ascii=False)
    
    print(f"\n统计结果已保存至: {stats_path}")
    
    # 汇总
    print("\n" + "=" * 60)
    print("数据集汇总")
    print("=" * 60)
    print(f"类别: {list(CLASSES.values())[1:]}")  # 不含背景
    print(f"训练集: {all_stats['train']['detection']['total_labels']} 图像")
    print(f"验证集: {all_stats['val']['detection']['total_labels']} 图像")
    print(f"测试集: {all_stats['test']['detection']['total_labels']} 图像")


if __name__ == "__main__":
    main()
