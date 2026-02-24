"""
可视化工具函数
用于绘制检测框、分割mask、对比图等
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Optional
import os


# 默认颜色方案 (BGR格式)
DEFAULT_COLORS = {
    1: (0, 255, 0),    # Scratch - 绿色
    2: (0, 165, 255),  # Spot - 橙色
    3: (0, 0, 255),    # Rust - 红色
}

CLASS_NAMES = {
    1: "Scratch",
    2: "Spot",
    3: "Rust",
}


def draw_bbox(
    image: np.ndarray,
    bbox: List[float],
    class_id: int,
    score: float = None,
    color: Tuple[int, int, int] = None,
    thickness: int = 2
) -> np.ndarray:
    """
    在图像上绘制边界框
    
    Args:
        image: BGR图像
        bbox: [x, y, width, height]
        class_id: 类别ID
        score: 置信度分数
        color: BGR颜色
        thickness: 线宽
    
    Returns:
        绘制后的图像
    """
    img = image.copy()
    
    if color is None:
        color = DEFAULT_COLORS.get(class_id, (0, 255, 0))
    
    x, y, w, h = [int(v) for v in bbox]
    
    # 绘制边界框
    cv2.rectangle(img, (x, y), (x + w, y + h), color, thickness)
    
    # 绘制标签
    label = CLASS_NAMES.get(class_id, f"Class {class_id}")
    if score is not None:
        label = f"{label}: {score:.2f}"
    
    # 标签背景
    (label_w, label_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    cv2.rectangle(img, (x, y - label_h - 10), (x + label_w, y), color, -1)
    cv2.putText(img, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    return img


def draw_mask(
    image: np.ndarray,
    mask: np.ndarray,
    class_id: int,
    color: Tuple[int, int, int] = None,
    alpha: float = 0.5
) -> np.ndarray:
    """
    在图像上绘制分割mask
    
    Args:
        image: BGR图像
        mask: 二值mask
        class_id: 类别ID
        color: BGR颜色
        alpha: 透明度
    
    Returns:
        绘制后的图像
    """
    img = image.copy()
    
    if color is None:
        color = DEFAULT_COLORS.get(class_id, (0, 255, 0))
    
    # 创建彩色mask
    colored_mask = np.zeros_like(img)
    colored_mask[mask > 0] = color
    
    # 叠加
    img = cv2.addWeighted(img, 1, colored_mask, alpha, 0)
    
    # 绘制轮廓
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(img, contours, -1, color, 2)
    
    return img


def visualize_detection_results(
    image: np.ndarray,
    predictions: List[Dict],
    score_threshold: float = 0.5
) -> np.ndarray:
    """
    可视化检测结果
    
    Args:
        image: BGR图像
        predictions: 预测结果列表
            [{"bbox": [x,y,w,h], "category_id": int, "score": float}, ...]
        score_threshold: 置信度阈值
    
    Returns:
        可视化后的图像
    """
    img = image.copy()
    
    for pred in predictions:
        if pred["score"] < score_threshold:
            continue
        
        img = draw_bbox(
            img,
            pred["bbox"],
            pred["category_id"],
            pred["score"]
        )
    
    return img


def visualize_segmentation_results(
    image: np.ndarray,
    predictions: List[Dict],
    score_threshold: float = 0.5,
    alpha: float = 0.5
) -> np.ndarray:
    """
    可视化分割结果
    
    Args:
        image: BGR图像
        predictions: 预测结果列表
            [{"mask": np.ndarray, "category_id": int, "score": float}, ...]
        score_threshold: 置信度阈值
        alpha: 透明度
    
    Returns:
        可视化后的图像
    """
    img = image.copy()
    
    for pred in predictions:
        if pred["score"] < score_threshold:
            continue
        
        img = draw_mask(
            img,
            pred["mask"],
            pred["category_id"],
            alpha=alpha
        )
    
    return img


def visualize_instance_results(
    image: np.ndarray,
    predictions: List[Dict],
    score_threshold: float = 0.5,
    show_bbox: bool = True,
    show_mask: bool = True,
    alpha: float = 0.5
) -> np.ndarray:
    """
    可视化实例分割结果（bbox + mask）
    
    Args:
        image: BGR图像
        predictions: 预测结果列表
        score_threshold: 置信度阈值
        show_bbox: 是否显示边界框
        show_mask: 是否显示mask
        alpha: mask透明度
    
    Returns:
        可视化后的图像
    """
    img = image.copy()
    
    for pred in predictions:
        if pred["score"] < score_threshold:
            continue
        
        if show_mask and "mask" in pred:
            img = draw_mask(img, pred["mask"], pred["category_id"], alpha=alpha)
        
        if show_bbox and "bbox" in pred:
            img = draw_bbox(img, pred["bbox"], pred["category_id"], pred["score"])
    
    return img


def visualize_gt_vs_pred(
    image: np.ndarray,
    gt_annotations: List[Dict],
    predictions: List[Dict],
    score_threshold: float = 0.5
) -> np.ndarray:
    """
    可视化GT与预测结果对比
    
    Args:
        image: BGR图像
        gt_annotations: GT标注列表
        predictions: 预测结果列表
        score_threshold: 置信度阈值
    
    Returns:
        左右对比图像
    """
    h, w = image.shape[:2]
    
    # GT图像
    gt_img = image.copy()
    for gt in gt_annotations:
        if "mask" in gt:
            gt_img = draw_mask(gt_img, gt["mask"], gt["category_id"], alpha=0.5)
        if "bbox" in gt:
            gt_img = draw_bbox(gt_img, gt["bbox"], gt["category_id"])
    
    # 预测图像
    pred_img = image.copy()
    for pred in predictions:
        if pred["score"] < score_threshold:
            continue
        if "mask" in pred:
            pred_img = draw_mask(pred_img, pred["mask"], pred["category_id"], alpha=0.5)
        if "bbox" in pred:
            pred_img = draw_bbox(pred_img, pred["bbox"], pred["category_id"], pred["score"])
    
    # 添加标题
    gt_img = cv2.copyMakeBorder(gt_img, 40, 0, 0, 0, cv2.BORDER_CONSTANT, value=(255, 255, 255))
    pred_img = cv2.copyMakeBorder(pred_img, 40, 0, 0, 0, cv2.BORDER_CONSTANT, value=(255, 255, 255))
    
    cv2.putText(gt_img, "Ground Truth", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    cv2.putText(pred_img, "Prediction", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    
    # 拼接
    comparison = np.hstack([gt_img, pred_img])
    
    return comparison


def save_visualization(
    image: np.ndarray,
    output_path: str
) -> None:
    """
    保存可视化结果
    
    Args:
        image: 要保存的图像
        output_path: 输出路径
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, image)


def create_legend(
    class_names: Dict[int, str],
    colors: Dict[int, Tuple[int, int, int]],
    output_path: str = None
) -> np.ndarray:
    """
    创建图例
    
    Args:
        class_names: 类别名称字典
        colors: 类别颜色字典
        output_path: 可选的保存路径
    
    Returns:
        图例图像
    """
    n_classes = len(class_names)
    legend = np.ones((50 * n_classes + 20, 200, 3), dtype=np.uint8) * 255
    
    for i, (class_id, name) in enumerate(class_names.items()):
        y = 30 + i * 50
        color = colors.get(class_id, (0, 0, 0))
        
        # 绘制颜色块
        cv2.rectangle(legend, (10, y - 15), (40, y + 15), color, -1)
        cv2.rectangle(legend, (10, y - 15), (40, y + 15), (0, 0, 0), 1)
        
        # 绘制类别名
        cv2.putText(legend, name, (50, y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
    
    if output_path:
        save_visualization(legend, output_path)
    
    return legend


def plot_metrics_curve(
    metrics_history: Dict[str, List[float]],
    output_path: str,
    title: str = "Training Metrics"
) -> None:
    """
    绘制指标曲线
    
    Args:
        metrics_history: 指标历史记录 {"loss": [...], "mAP": [...]}
        output_path: 保存路径
        title: 图表标题
    """
    plt.figure(figsize=(12, 4))
    
    n_metrics = len(metrics_history)
    for i, (name, values) in enumerate(metrics_history.items()):
        plt.subplot(1, n_metrics, i + 1)
        plt.plot(values)
        plt.title(name)
        plt.xlabel("Epoch")
        plt.ylabel(name)
        plt.grid(True)
    
    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def visualize_dataset_statistics(
    stats: Dict,
    output_dir: str
) -> None:
    """
    可视化数据集统计信息
    
    Args:
        stats: 统计信息字典
        output_dir: 输出目录
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 类别分布
    if "class_distribution" in stats:
        plt.figure(figsize=(8, 6))
        classes = list(stats["class_distribution"].keys())
        counts = list(stats["class_distribution"].values())
        plt.bar(classes, counts, color=['green', 'red', 'blue'][:len(classes)])
        plt.xlabel("Class")
        plt.ylabel("Count")
        plt.title("Class Distribution")
        plt.savefig(os.path.join(output_dir, "class_distribution.png"), dpi=150)
        plt.close()
    
    # 目标尺寸分布
    if "size_distribution" in stats:
        plt.figure(figsize=(10, 4))
        
        plt.subplot(1, 2, 1)
        plt.hist(stats["size_distribution"]["widths"], bins=50, alpha=0.7)
        plt.xlabel("Width (pixels)")
        plt.ylabel("Count")
        plt.title("Width Distribution")
        
        plt.subplot(1, 2, 2)
        plt.hist(stats["size_distribution"]["heights"], bins=50, alpha=0.7)
        plt.xlabel("Height (pixels)")
        plt.ylabel("Count")
        plt.title("Height Distribution")
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "size_distribution.png"), dpi=150)
        plt.close()
