"""
Mask处理工具函数
用于处理分割掩膜、提取轮廓、转换格式等
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional, Dict
from scipy import ndimage


def load_mask(mask_path: str) -> np.ndarray:
    """
    加载mask图像
    
    Args:
        mask_path: mask文件路径
    
    Returns:
        单通道mask数组
    """
    mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
    
    # 如果是3通道，取第一个通道
    if len(mask.shape) == 3:
        mask = mask[:, :, 0]
    
    return mask


def extract_instances_from_mask(
    mask: np.ndarray,
    class_id: int
) -> List[np.ndarray]:
    """
    从语义分割mask中提取特定类别的所有实例
    使用连通域分析区分不同实例
    
    Args:
        mask: 语义分割mask (H, W)，像素值为类别ID
        class_id: 要提取的类别ID
    
    Returns:
        实例mask列表，每个为二值mask
    """
    # 创建该类别的二值mask
    binary_mask = (mask == class_id).astype(np.uint8)
    
    # 连通域分析
    labeled_array, num_features = ndimage.label(binary_mask)
    
    instances = []
    for i in range(1, num_features + 1):
        instance_mask = (labeled_array == i).astype(np.uint8)
        # 过滤太小的区域（噪声）
        if instance_mask.sum() > 10:  # 最小像素数阈值
            instances.append(instance_mask)
    
    return instances


def mask_to_polygon(binary_mask: np.ndarray, tolerance: float = 1.0) -> List[List[float]]:
    """
    将二值mask转换为多边形表示
    
    Args:
        binary_mask: 二值mask (H, W)
        tolerance: 轮廓简化容差
    
    Returns:
        多边形列表 [[x1,y1,x2,y2,...], ...]
    """
    # 找轮廓
    contours, _ = cv2.findContours(
        binary_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )
    
    polygons = []
    for contour in contours:
        # 简化轮廓
        epsilon = tolerance
        approx = cv2.approxPolyDP(contour, epsilon, True)
        
        # 至少需要3个点构成多边形
        if len(approx) >= 3:
            # 转换为COCO格式 [x1,y1,x2,y2,...]
            polygon = approx.flatten().tolist()
            if len(polygon) >= 6:  # 至少3个点
                polygons.append(polygon)
    
    return polygons


def mask_to_bbox(binary_mask: np.ndarray) -> Tuple[List[float], float]:
    """
    从二值mask计算边界框
    
    Args:
        binary_mask: 二值mask (H, W)
    
    Returns:
        (bbox, area): [x, y, width, height] 和面积
    """
    rows = np.any(binary_mask, axis=1)
    cols = np.any(binary_mask, axis=0)
    
    if not rows.any() or not cols.any():
        return [0, 0, 0, 0], 0
    
    y_min, y_max = np.where(rows)[0][[0, -1]]
    x_min, x_max = np.where(cols)[0][[0, -1]]
    
    width = x_max - x_min + 1
    height = y_max - y_min + 1
    area = float(binary_mask.sum())
    
    return [float(x_min), float(y_min), float(width), float(height)], area


def compute_mask_iou(mask1: np.ndarray, mask2: np.ndarray) -> float:
    """
    计算两个mask的IoU
    
    Args:
        mask1: 第一个二值mask
        mask2: 第二个二值mask
    
    Returns:
        IoU值
    """
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    
    if union == 0:
        return 0.0
    
    return intersection / union


def resize_mask(mask: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
    """
    调整mask大小
    
    Args:
        mask: 输入mask
        target_size: 目标尺寸 (width, height)
    
    Returns:
        调整后的mask
    """
    return cv2.resize(mask, target_size, interpolation=cv2.INTER_NEAREST)


def visualize_mask(
    image: np.ndarray,
    mask: np.ndarray,
    class_colors: Dict[int, Tuple[int, int, int]],
    alpha: float = 0.5
) -> np.ndarray:
    """
    在图像上可视化mask
    
    Args:
        image: BGR图像
        mask: 语义分割mask
        class_colors: 类别ID到颜色的映射
        alpha: 透明度
    
    Returns:
        叠加后的图像
    """
    overlay = image.copy()
    
    for class_id, color in class_colors.items():
        if class_id == 0:  # 跳过背景
            continue
        class_mask = mask == class_id
        overlay[class_mask] = color
    
    result = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)
    return result


def get_instance_info(
    mask: np.ndarray,
    num_classes: int
) -> Dict[int, int]:
    """
    获取mask中每个类别的实例数量
    
    Args:
        mask: 语义分割mask
        num_classes: 类别数量（不含背景）
    
    Returns:
        {class_id: instance_count}
    """
    info = {}
    for class_id in range(1, num_classes + 1):
        instances = extract_instances_from_mask(mask, class_id)
        info[class_id] = len(instances)
    return info
