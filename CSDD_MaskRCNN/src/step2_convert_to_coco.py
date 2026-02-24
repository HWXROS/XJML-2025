#!/usr/bin/env python3
"""
第2步：数据格式转换
将CSDD数据集从YOLO格式+分割掩膜转换为COCO格式

输入:
- CSDD_det: YOLO格式检测标注 (txt)
- CSDD_seg: 分割掩膜 (png)

输出:
- COCO格式JSON文件 (包含bbox和segmentation)
- 整理后的图像文件夹
"""

import os
import sys
import cv2
import json
import shutil
import numpy as np
import gc
from collections import defaultdict
from tqdm import tqdm
from typing import List, Dict, Tuple

# 添加项目路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.coco_utils import (
    create_coco_structure,
    add_image_to_coco,
    add_annotation_to_coco,
    save_coco_json
)
from utils.mask_utils import (
    load_mask,
    extract_instances_from_mask,
    mask_to_polygon,
    mask_to_bbox
)


# ======================== 配置 ========================
# 输入路径 - 根据你的实际路径修改
CSDD_DET_ROOT = "data/CSDD_raw/CSDD_det"
CSDD_SEG_ROOT = "data/CSDD_raw/CSDD_seg"

# 输出路径
OUTPUT_ROOT = "data/CSDD_coco"

# 类别定义 (mask像素值 -> COCO类别ID)
# mask中: 0=背景, 1=Scratch, 2=Spot, 3=Rust
# COCO中: 类别ID从1开始
CLASSES = ["Scratch", "Spot", "Rust"]  # 对应mask值1, 2, 3
NUM_CLASSES = len(CLASSES)

# 类别映射: YOLO类别ID -> COCO类别ID
# YOLO: 0=Scratch, 1=Spot, 2=Rust
# COCO: 1=Scratch, 2=Spot, 3=Rust
YOLO_TO_COCO_CLASS = {
    0: 1,  # YOLO类别0 -> COCO类别1 (Scratch)
    1: 2,  # YOLO类别1 -> COCO类别2 (Spot)
    2: 3,  # YOLO类别2 -> COCO类别3 (Rust)
}

# 数据集划分映射
SPLITS = {
    "train": {"det": "train2017", "seg": "train"},
    "val": {"det": "val2017", "seg": "val"},
    "test": {"det": "test2017", "seg": "test"}
}


def get_image_info(image_path: str) -> Tuple[int, int]:
    """获取图像尺寸"""
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"无法读取图像: {image_path}")
    return img.shape[1], img.shape[0]


def parse_yolo_label(label_path: str) -> List[Tuple[int, List[float]]]:
    """解析YOLO格式标注"""
    annotations = []
    if not os.path.exists(label_path):
        return annotations
    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                class_id = int(parts[0])
                bbox = [float(x) for x in parts[1:5]]
                annotations.append((class_id, bbox))
    return annotations


def yolo_bbox_to_coco(yolo_bbox: List[float], img_width: int, img_height: int) -> Tuple[List[float], float]:
    """将YOLO归一化bbox转换为COCO绝对坐标bbox"""
    x_center, y_center, w, h = yolo_bbox
    x_center_px = x_center * img_width
    y_center_px = y_center * img_height
    w_px = w * img_width
    h_px = h * img_height
    x_min = max(0, x_center_px - w_px / 2)
    y_min = max(0, y_center_px - h_px / 2)
    w_px = min(w_px, img_width - x_min)
    h_px = min(h_px, img_height - y_min)
    area = w_px * h_px
    return [round(x_min, 2), round(y_min, 2), round(w_px, 2), round(h_px, 2)], area


def match_bbox_to_instance(yolo_bbox: List[float], instances: List[np.ndarray], 
                           img_width: int, img_height: int, iou_threshold: float = 0.3) -> int:
    """将YOLO bbox匹配到最佳的mask实例"""
    coco_bbox, _ = yolo_bbox_to_coco(yolo_bbox, img_width, img_height)
    x1, y1, w, h = coco_bbox
    x2, y2 = x1 + w, y1 + h
    
    best_iou = 0
    best_idx = -1
    
    for idx, inst_mask in enumerate(instances):
        inst_bbox, _ = mask_to_bbox(inst_mask)
        ix1, iy1, iw, ih = inst_bbox
        ix2, iy2 = ix1 + iw, iy1 + ih
        
        inter_x1 = max(x1, ix1)
        inter_y1 = max(y1, iy1)
        inter_x2 = min(x2, ix2)
        inter_y2 = min(y2, iy2)
        
        inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
        union_area = w * h + iw * ih - inter_area
        
        if union_area > 0:
            iou = inter_area / union_area
            if iou > best_iou:
                best_iou = iou
                best_idx = idx
    
    return best_idx if best_iou >= iou_threshold else -1


def process_images(image_files, det_images_dir, output_images_dir):
    """处理图像信息并复制图像文件"""
    image_info = []
    for image_file in tqdm(image_files, desc="处理图像信息"):
        image_path = os.path.join(det_images_dir, image_file)
        output_image_path = os.path.join(output_images_dir, image_file)
        
        try:
            img_width, img_height = get_image_info(image_path)
            # 复制图像文件
            if not os.path.exists(output_image_path):
                shutil.copy(image_path, output_image_path)
            image_info.append({
                "file_name": image_file,
                "width": img_width,
                "height": img_height
            })
        except Exception as e:
            print(f"[WARNING] 跳过图像 {image_file}: {e}")
    
    return image_info


def batch_process_images(image_files, det_images_dir, output_images_dir, batch_size=100):
    """分批处理图像信息"""
    all_image_info = []
    total_batches = (len(image_files) + batch_size - 1) // batch_size
    
    for batch_idx in range(total_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, len(image_files))
        batch_files = image_files[start_idx:end_idx]
        
        print(f"处理图像信息 - 批次 {batch_idx + 1}/{total_batches} ({start_idx}-{end_idx})")
        
        # 处理当前批次
        image_info_batch = process_images(batch_files, det_images_dir, output_images_dir)
        all_image_info.extend(image_info_batch)
        
        # 清理内存
        del image_info_batch
        gc.collect()
    
    return all_image_info


def process_detection_annotations(image_files, det_labels_dir):
    """处理检测标注"""
    detection_annotations = {}
    for image_file in tqdm(image_files, desc="处理检测标注"):
        image_name = os.path.splitext(image_file)[0]
        label_path = os.path.join(det_labels_dir, f"{image_name}.txt")
        
        if os.path.exists(label_path):
            yolo_annotations = parse_yolo_label(label_path)
            detection_annotations[image_file] = yolo_annotations
        else:
            detection_annotations[image_file] = []
    
    return detection_annotations


def batch_process_detection_annotations(image_files, det_labels_dir, batch_size=100):
    """分批处理检测标注"""
    all_detection_annotations = {}
    total_batches = (len(image_files) + batch_size - 1) // batch_size
    
    for batch_idx in range(total_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, len(image_files))
        batch_files = image_files[start_idx:end_idx]
        
        print(f"处理检测标注 - 批次 {batch_idx + 1}/{total_batches} ({start_idx}-{end_idx})")
        
        # 处理当前批次
        detection_annotations_batch = process_detection_annotations(batch_files, det_labels_dir)
        all_detection_annotations.update(detection_annotations_batch)
        
        # 清理内存
        del detection_annotations_batch
        gc.collect()
    
    return all_detection_annotations


def process_segmentation_masks(image_files, seg_masks_dir, num_classes):
    """处理分割掩膜"""
    segmentation_masks = {}
    for image_file in tqdm(image_files, desc="处理分割掩膜"):
        image_name = os.path.splitext(image_file)[0]
        mask_path = os.path.join(seg_masks_dir, f"{image_name}_mask.png")
        
        if os.path.exists(mask_path):
            mask = load_mask(mask_path)
            class_instances = {}
            for class_id in range(1, num_classes + 1):
                instances = extract_instances_from_mask(mask, class_id)
                class_instances[class_id] = instances
            segmentation_masks[image_file] = class_instances
            # 及时释放内存
            del mask
            gc.collect()
        else:
            segmentation_masks[image_file] = None
    
    return segmentation_masks


def batch_process_segmentation_masks(image_files, seg_masks_dir, num_classes, batch_size=20):
    """分批处理分割掩膜（减小批次大小以减少内存使用）"""
    all_segmentation_masks = {}
    total_batches = (len(image_files) + batch_size - 1) // batch_size
    
    for batch_idx in range(total_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, len(image_files))
        batch_files = image_files[start_idx:end_idx]
        
        print(f"处理分割掩膜 - 批次 {batch_idx + 1}/{total_batches} ({start_idx}-{end_idx})")
        
        # 处理当前批次
        segmentation_masks_batch = {}
        for image_file in tqdm(batch_files, desc="处理分割掩膜"):
            image_name = os.path.splitext(image_file)[0]
            mask_path = os.path.join(seg_masks_dir, f"{image_name}_mask.png")
            
            if os.path.exists(mask_path):
                mask = load_mask(mask_path)
                class_instances = {}
                for class_id in range(1, num_classes + 1):
                    instances = extract_instances_from_mask(mask, class_id)
                    class_instances[class_id] = instances
                segmentation_masks_batch[image_file] = class_instances
                # 及时释放内存
                del mask
                for class_id in class_instances:
                    del class_instances[class_id]
                gc.collect()
            else:
                segmentation_masks_batch[image_file] = None
        
        # 更新结果
        all_segmentation_masks.update(segmentation_masks_batch)
        
        # 清理内存
        del segmentation_masks_batch
        gc.collect()
    
    return all_segmentation_masks


def convert_split(split: str, det_root: str, seg_root: str, output_root: str) -> Dict:
    """转换一个数据集划分（逐张处理优化版）"""
    print(f"\n转换 {split.upper()} 集...")
    
    det_split = SPLITS[split]["det"]
    seg_split = SPLITS[split]["seg"]
    
    det_images_dir = os.path.join(det_root, "images", det_split)
    det_labels_dir = os.path.join(det_root, "labels", det_split)
    seg_masks_dir = os.path.join(seg_root, "ground_truth", seg_split)
    
    output_images_dir = os.path.join(output_root, split)
    output_ann_path = os.path.join(output_root, "annotations", f"instances_{split}.json")
    
    os.makedirs(output_images_dir, exist_ok=True)
    os.makedirs(os.path.dirname(output_ann_path), exist_ok=True)
    
    coco = create_coco_structure(CLASSES, f"CSDD {split} Dataset")
    
    stats = {
        "total_images": 0,
        "total_annotations": 0,
        "class_counts": defaultdict(int),
        "skipped_no_mask": 0,
        "skipped_no_match": 0
    }
    
    if not os.path.exists(det_images_dir):
        print(f"[ERROR] 图像目录不存在: {det_images_dir}")
        return stats
    
    image_files = sorted([f for f in os.listdir(det_images_dir) if f.endswith('.jpg')])
    
    image_id = 0
    ann_id = 0
    
    # 逐张处理图像，完全避免中间缓存
    for image_file in tqdm(image_files, desc=f"处理{split}"):
        image_name = os.path.splitext(image_file)[0]
        
        image_path = os.path.join(det_images_dir, image_file)
        label_path = os.path.join(det_labels_dir, f"{image_name}.txt")
        mask_path = os.path.join(seg_masks_dir, f"{image_name}_mask.png")
        
        try:
            # 1. 处理图像信息
            img_width, img_height = get_image_info(image_path)
            
            # 复制图像文件
            output_image_path = os.path.join(output_images_dir, image_file)
            if not os.path.exists(output_image_path):
                shutil.copy(image_path, output_image_path)
            
            # 2. 添加图像信息到COCO
            image_id += 1
            add_image_to_coco(coco, image_id, image_file, img_width, img_height)
            stats["total_images"] += 1
            
            # 3. 处理检测标注
            yolo_annotations = parse_yolo_label(label_path)
            
            # 4. 处理分割掩膜（逐张处理，及时释放内存）
            mask = None
            class_instances = {}
            
            if os.path.exists(mask_path):
                mask = load_mask(mask_path)
                # 提取每个类别的实例
                for class_id in range(1, NUM_CLASSES + 1):
                    instances = extract_instances_from_mask(mask, class_id)
                    class_instances[class_id] = instances
                # 及时释放mask内存
                del mask
                gc.collect()
            else:
                stats["skipped_no_mask"] += 1
            
            # 5. 生成标注
            used_instances = {c: set() for c in range(1, NUM_CLASSES + 1)}
            
            for yolo_class_id, yolo_bbox in yolo_annotations:
                coco_class_id = YOLO_TO_COCO_CLASS.get(yolo_class_id, yolo_class_id + 1)
                
                if coco_class_id > NUM_CLASSES:
                    continue
                
                coco_bbox, bbox_area = yolo_bbox_to_coco(yolo_bbox, img_width, img_height)
                
                segmentation = []
                area = bbox_area
                
                if class_instances and coco_class_id in class_instances:
                    available_instances = [
                        (i, inst) for i, inst in enumerate(class_instances[coco_class_id])
                        if i not in used_instances[coco_class_id]
                    ]
                    
                    if available_instances:
                        best_idx = match_bbox_to_instance(
                            yolo_bbox,
                            [inst for _, inst in available_instances],
                            img_width, img_height
                        )
                        
                        if best_idx >= 0:
                            orig_idx, inst_mask = available_instances[best_idx]
                            used_instances[coco_class_id].add(orig_idx)
                            
                            polygons = mask_to_polygon(inst_mask)
                            if polygons:
                                segmentation = polygons
                                area = float(inst_mask.sum())
                            
                            # 及时释放实例掩码内存
                            del inst_mask
                            gc.collect()
                
                if not segmentation:
                    x, y, w, h = coco_bbox
                    segmentation = [[x, y, x+w, y, x+w, y+h, x, y+h]]
                    stats["skipped_no_match"] += 1
                
                ann_id += 1
                add_annotation_to_coco(
                    coco,
                    ann_id=ann_id,
                    image_id=image_id,
                    category_id=coco_class_id,
                    bbox=coco_bbox,
                    segmentation=segmentation,
                    area=area,
                    iscrowd=0
                )
                
                stats["total_annotations"] += 1
                stats["class_counts"][coco_class_id] += 1
            
            # 及时释放内存
            del yolo_annotations
            # 释放class_instances中的所有实例
            for class_id in list(class_instances.keys()):
                del class_instances[class_id]
            del class_instances
            gc.collect()
            
        except Exception as e:
            print(f"[WARNING] 跳过图像 {image_file}: {e}")
            # 及时释放内存
            gc.collect()
            continue
    
    save_coco_json(coco, output_ann_path)
    return stats


def verify_coco_format(json_path: str) -> bool:
    """验证COCO格式是否正确"""
    try:
        from pycocotools.coco import COCO
        coco = COCO(json_path)
        
        print(f"  图像数量: {len(coco.imgs)}")
        print(f"  标注数量: {len(coco.anns)}")
        print(f"  类别数量: {len(coco.cats)}")
        
        for cat_id, cat_info in coco.cats.items():
            ann_ids = coco.getAnnIds(catIds=[cat_id])
            print(f"  - {cat_info['name']}: {len(ann_ids)} 个标注")
        
        return True
    except Exception as e:
        print(f"[ERROR] COCO格式验证失败: {e}")
        return False


def main():
    """主函数"""
    print("=" * 60)
    print("CSDD 数据格式转换: YOLO + Mask -> COCO")
    print("=" * 60)
    
    if not os.path.exists(CSDD_DET_ROOT):
        print(f"[ERROR] 检测数据目录不存在: {CSDD_DET_ROOT}")
        print("请修改 CSDD_DET_ROOT 变量为正确的路径")
        return
    
    if not os.path.exists(CSDD_SEG_ROOT):
        print(f"[ERROR] 分割数据目录不存在: {CSDD_SEG_ROOT}")
        print("请修改 CSDD_SEG_ROOT 变量为正确的路径")
        return
    
    print(f"\n输入路径:")
    print(f"  检测数据: {CSDD_DET_ROOT}")
    print(f"  分割数据: {CSDD_SEG_ROOT}")
    print(f"输出路径: {OUTPUT_ROOT}")
    print(f"类别: {CLASSES}")
    
    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    
    all_stats = {}
    for split in ["train", "val", "test"]:
        stats = convert_split(split, CSDD_DET_ROOT, CSDD_SEG_ROOT, OUTPUT_ROOT)
        all_stats[split] = stats
        
        print(f"\n{split.upper()} 统计:")
        print(f"  图像: {stats['total_images']}")
        print(f"  标注: {stats['total_annotations']}")
        print(f"  各类别: {dict(stats['class_counts'])}")
    
    print("\n" + "=" * 60)
    print("验证COCO格式")
    print("=" * 60)
    
    for split in ["train", "val", "test"]:
        json_path = os.path.join(OUTPUT_ROOT, "annotations", f"instances_{split}.json")
        print(f"\n{split.upper()}:")
        verify_coco_format(json_path)
    
    print("\n" + "=" * 60)
    print("转换完成!")
    print("=" * 60)


if __name__ == "__main__":
    main()
