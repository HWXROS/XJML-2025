"""
评估指标计算工具
用于计算mAP、IoU、mIoU等指标
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from collections import defaultdict


def compute_iou_bbox(box1: List[float], box2: List[float]) -> float:
    """
    计算两个bbox的IoU
    
    Args:
        box1: [x1, y1, w1, h1]
        box2: [x2, y2, w2, h2]
    
    Returns:
        IoU值
    """
    # 转换为 [x1, y1, x2, y2] 格式
    box1_xyxy = [box1[0], box1[1], box1[0] + box1[2], box1[1] + box1[3]]
    box2_xyxy = [box2[0], box2[1], box2[0] + box2[2], box2[1] + box2[3]]
    
    # 计算交集
    x1 = max(box1_xyxy[0], box2_xyxy[0])
    y1 = max(box1_xyxy[1], box2_xyxy[1])
    x2 = min(box1_xyxy[2], box2_xyxy[2])
    y2 = min(box1_xyxy[3], box2_xyxy[3])
    
    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    
    # 计算并集
    box1_area = box1[2] * box1[3]
    box2_area = box2[2] * box2[3]
    union_area = box1_area + box2_area - inter_area
    
    if union_area == 0:
        return 0.0
    
    return inter_area / union_area


def compute_iou_mask(mask1: np.ndarray, mask2: np.ndarray) -> float:
    """
    计算两个mask的IoU
    
    Args:
        mask1: 二值mask1
        mask2: 二值mask2
    
    Returns:
        IoU值
    """
    intersection = np.logical_and(mask1 > 0, mask2 > 0).sum()
    union = np.logical_or(mask1 > 0, mask2 > 0).sum()
    
    if union == 0:
        return 0.0
    
    return intersection / union


def compute_dice(mask1: np.ndarray, mask2: np.ndarray) -> float:
    """
    计算Dice系数
    
    Args:
        mask1: 二值mask1
        mask2: 二值mask2
    
    Returns:
        Dice系数
    """
    intersection = np.logical_and(mask1 > 0, mask2 > 0).sum()
    total = (mask1 > 0).sum() + (mask2 > 0).sum()
    
    if total == 0:
        return 0.0
    
    return 2 * intersection / total


def compute_ap(
    recalls: np.ndarray,
    precisions: np.ndarray
) -> float:
    """
    计算Average Precision (AP)
    使用11点插值或全点插值
    
    Args:
        recalls: 召回率数组
        precisions: 精确率数组
    
    Returns:
        AP值
    """
    # 添加起始和结束点
    recalls = np.concatenate([[0], recalls, [1]])
    precisions = np.concatenate([[0], precisions, [0]])
    
    # 使精确率单调递减
    for i in range(len(precisions) - 2, -1, -1):
        precisions[i] = max(precisions[i], precisions[i + 1])
    
    # 找到召回率变化的点
    recall_change = np.where(recalls[1:] != recalls[:-1])[0]
    
    # 计算AP
    ap = np.sum((recalls[recall_change + 1] - recalls[recall_change]) * precisions[recall_change + 1])
    
    return ap


def evaluate_detection(
    predictions: List[Dict],
    ground_truths: List[Dict],
    iou_thresholds: List[float] = None,
    num_classes: int = 2
) -> Dict:
    """
    评估检测性能
    
    Args:
        predictions: 预测结果列表
            [{"image_id": int, "category_id": int, "bbox": [x,y,w,h], "score": float}, ...]
        ground_truths: 真值列表
            [{"image_id": int, "category_id": int, "bbox": [x,y,w,h]}, ...]
        iou_thresholds: IoU阈值列表
        num_classes: 类别数量
    
    Returns:
        评估结果字典
    """
    if iou_thresholds is None:
        iou_thresholds = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
    
    results = {
        "AP": {},
        "mAP": {},
        "per_class_AP": defaultdict(dict)
    }
    
    # 按类别组织预测和真值
    for iou_thr in iou_thresholds:
        class_aps = []
        
        for class_id in range(1, num_classes + 1):
            # 筛选当前类别
            class_preds = [p for p in predictions if p["category_id"] == class_id]
            class_gts = [g for g in ground_truths if g["category_id"] == class_id]
            
            if len(class_gts) == 0:
                continue
            
            # 按置信度排序
            class_preds = sorted(class_preds, key=lambda x: x["score"], reverse=True)
            
            # 计算TP/FP
            tp = np.zeros(len(class_preds))
            fp = np.zeros(len(class_preds))
            gt_matched = defaultdict(set)
            
            for pred_idx, pred in enumerate(class_preds):
                img_id = pred["image_id"]
                img_gts = [g for g in class_gts if g["image_id"] == img_id]
                
                best_iou = 0
                best_gt_idx = -1
                
                for gt_idx, gt in enumerate(img_gts):
                    if gt_idx in gt_matched[img_id]:
                        continue
                    iou = compute_iou_bbox(pred["bbox"], gt["bbox"])
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = gt_idx
                
                if best_iou >= iou_thr:
                    tp[pred_idx] = 1
                    gt_matched[img_id].add(best_gt_idx)
                else:
                    fp[pred_idx] = 1
            
            # 计算precision和recall
            tp_cumsum = np.cumsum(tp)
            fp_cumsum = np.cumsum(fp)
            
            recalls = tp_cumsum / len(class_gts)
            precisions = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-10)
            
            # 计算AP
            ap = compute_ap(recalls, precisions)
            class_aps.append(ap)
            results["per_class_AP"][class_id][f"AP@{iou_thr}"] = ap
        
        # 计算mAP
        if class_aps:
            results["mAP"][f"mAP@{iou_thr}"] = np.mean(class_aps)
    
    # 计算COCO风格的mAP (AP@[0.5:0.95])
    if iou_thresholds == [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]:
        results["mAP"]["mAP@[.5:.95]"] = np.mean(list(results["mAP"].values()))
    
    return results


def evaluate_segmentation(
    pred_masks: List[Dict],
    gt_masks: List[Dict],
    num_classes: int = 2
) -> Dict:
    """
    评估分割性能
    
    Args:
        pred_masks: 预测mask列表
            [{"image_id": int, "category_id": int, "mask": np.ndarray, "score": float}, ...]
        gt_masks: 真值mask列表
            [{"image_id": int, "category_id": int, "mask": np.ndarray}, ...]
        num_classes: 类别数量
    
    Returns:
        评估结果字典
    """
    results = {
        "IoU": defaultdict(list),
        "Dice": defaultdict(list),
        "per_class": {}
    }
    
    # 按图像ID组织
    gt_by_image = defaultdict(list)
    for gt in gt_masks:
        gt_by_image[gt["image_id"]].append(gt)
    
    pred_by_image = defaultdict(list)
    for pred in pred_masks:
        pred_by_image[pred["image_id"]].append(pred)
    
    # 对每个图像评估
    for img_id in gt_by_image.keys():
        img_gts = gt_by_image[img_id]
        img_preds = pred_by_image.get(img_id, [])
        
        # 按类别评估
        for class_id in range(1, num_classes + 1):
            class_gts = [g for g in img_gts if g["category_id"] == class_id]
            class_preds = [p for p in img_preds if p["category_id"] == class_id]
            
            if not class_gts:
                continue
            
            # 对每个GT找最佳匹配的预测
            gt_matched = set()
            pred_matched = set()
            
            for gt_idx, gt in enumerate(class_gts):
                best_iou = 0
                best_pred_idx = -1
                
                for pred_idx, pred in enumerate(class_preds):
                    if pred_idx in pred_matched:
                        continue
                    iou = compute_iou_mask(pred["mask"], gt["mask"])
                    if iou > best_iou:
                        best_iou = iou
                        best_pred_idx = pred_idx
                
                if best_pred_idx >= 0 and best_iou > 0.5:  # IoU阈值
                    results["IoU"][class_id].append(best_iou)
                    dice = compute_dice(class_preds[best_pred_idx]["mask"], gt["mask"])
                    results["Dice"][class_id].append(dice)
                    pred_matched.add(best_pred_idx)
                else:
                    results["IoU"][class_id].append(0)
                    results["Dice"][class_id].append(0)
    
    # 计算每类平均
    for class_id in range(1, num_classes + 1):
        if results["IoU"][class_id]:
            results["per_class"][class_id] = {
                "IoU": np.mean(results["IoU"][class_id]),
                "Dice": np.mean(results["Dice"][class_id])
            }
    
    # 计算mIoU
    all_ious = []
    all_dices = []
    for class_id in range(1, num_classes + 1):
        if class_id in results["per_class"]:
            all_ious.append(results["per_class"][class_id]["IoU"])
            all_dices.append(results["per_class"][class_id]["Dice"])
    
    results["mIoU"] = np.mean(all_ious) if all_ious else 0
    results["mDice"] = np.mean(all_dices) if all_dices else 0
    
    return results


def print_detection_results(results: Dict, class_names: List[str]) -> None:
    """
    打印检测评估结果
    
    Args:
        results: evaluate_detection的返回结果
        class_names: 类别名称列表
    """
    print("\n" + "=" * 60)
    print("检测评估结果 (Detection Results)")
    print("=" * 60)
    
    # 打印每个类别的AP
    print(f"\n{'类别':<15} {'AP@0.5':<12} {'AP@0.75':<12} {'AP@[.5:.95]':<12}")
    print("-" * 50)
    
    for class_id, class_name in enumerate(class_names, 1):
        if class_id in results["per_class_AP"]:
            ap50 = results["per_class_AP"][class_id].get("AP@0.5", 0)
            ap75 = results["per_class_AP"][class_id].get("AP@0.75", 0)
            ap_all = np.mean([v for k, v in results["per_class_AP"][class_id].items()])
            print(f"{class_name:<15} {ap50*100:>10.2f}% {ap75*100:>10.2f}% {ap_all*100:>10.2f}%")
    
    print("-" * 50)
    print(f"{'mAP':<15} {results['mAP'].get('mAP@0.5', 0)*100:>10.2f}% "
          f"{results['mAP'].get('mAP@0.75', 0)*100:>10.2f}% "
          f"{results['mAP'].get('mAP@[.5:.95]', 0)*100:>10.2f}%")


def print_segmentation_results(results: Dict, class_names: List[str]) -> None:
    """
    打印分割评估结果
    
    Args:
        results: evaluate_segmentation的返回结果
        class_names: 类别名称列表
    """
    print("\n" + "=" * 60)
    print("分割评估结果 (Segmentation Results)")
    print("=" * 60)
    
    print(f"\n{'类别':<15} {'IoU':<12} {'Dice':<12}")
    print("-" * 40)
    
    for class_id, class_name in enumerate(class_names, 1):
        if class_id in results["per_class"]:
            iou = results["per_class"][class_id]["IoU"]
            dice = results["per_class"][class_id]["Dice"]
            print(f"{class_name:<15} {iou*100:>10.2f}% {dice*100:>10.2f}%")
    
    print("-" * 40)
    print(f"{'mIoU':<15} {results['mIoU']*100:>10.2f}%")
    print(f"{'mDice':<15} {results['mDice']*100:>10.2f}%")
