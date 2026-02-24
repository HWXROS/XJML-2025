"""
COCO格式工具函数
用于创建和操作COCO格式的标注文件
"""

import json
import os
from typing import Dict, List, Tuple, Optional
from datetime import datetime


def create_coco_structure(
    classes: List[str],
    description: str = "CSDD Dataset"
) -> Dict:
    """
    创建COCO格式的基础结构
    
    Args:
        classes: 类别名称列表 (不包含背景)
        description: 数据集描述
    
    Returns:
        COCO格式的字典
    """
    coco = {
        "info": {
            "description": description,
            "version": "1.0",
            "year": datetime.now().year,
            "contributor": "CSDD",
            "date_created": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        },
        "licenses": [
            {
                "id": 1,
                "name": "Unknown",
                "url": ""
            }
        ],
        "images": [],
        "annotations": [],
        "categories": []
    }
    
    # 添加类别 (从1开始，0为背景)
    for i, class_name in enumerate(classes):
        coco["categories"].append({
            "id": i + 1,  # COCO类别ID从1开始
            "name": class_name,
            "supercategory": "defect"
        })
    
    return coco


def add_image_to_coco(
    coco: Dict,
    image_id: int,
    file_name: str,
    width: int,
    height: int
) -> None:
    """
    向COCO结构添加图像信息
    
    Args:
        coco: COCO字典
        image_id: 图像ID
        file_name: 文件名
        width: 图像宽度
        height: 图像高度
    """
    coco["images"].append({
        "id": image_id,
        "file_name": file_name,
        "width": width,
        "height": height,
        "license": 1,
        "date_captured": ""
    })


def add_annotation_to_coco(
    coco: Dict,
    ann_id: int,
    image_id: int,
    category_id: int,
    bbox: List[float],
    segmentation: List[List[float]],
    area: float,
    iscrowd: int = 0
) -> None:
    """
    向COCO结构添加标注信息
    
    Args:
        coco: COCO字典
        ann_id: 标注ID
        image_id: 图像ID
        category_id: 类别ID (从1开始)
        bbox: 边界框 [x, y, width, height]
        segmentation: 分割多边形 [[x1,y1,x2,y2,...], ...]
        area: 面积
        iscrowd: 是否为crowd标注
    """
    coco["annotations"].append({
        "id": ann_id,
        "image_id": image_id,
        "category_id": category_id,
        "bbox": bbox,
        "segmentation": segmentation,
        "area": area,
        "iscrowd": iscrowd
    })


def save_coco_json(coco: Dict, output_path: str) -> None:
    """
    保存COCO格式JSON文件
    
    Args:
        coco: COCO字典
        output_path: 输出路径
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(coco, f, ensure_ascii=False, indent=2)
    print(f"[INFO] 保存COCO标注: {output_path}")
    print(f"       图像数量: {len(coco['images'])}")
    print(f"       标注数量: {len(coco['annotations'])}")


def load_coco_json(json_path: str) -> Dict:
    """
    加载COCO格式JSON文件
    
    Args:
        json_path: JSON文件路径
    
    Returns:
        COCO字典
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def yolo_to_coco_bbox(
    yolo_bbox: List[float],
    img_width: int,
    img_height: int
) -> Tuple[List[float], float]:
    """
    将YOLO格式bbox转换为COCO格式
    
    YOLO格式: [class_id, x_center, y_center, width, height] (归一化)
    COCO格式: [x_min, y_min, width, height] (像素坐标)
    
    Args:
        yolo_bbox: YOLO格式的bbox [x_center, y_center, width, height]
        img_width: 图像宽度
        img_height: 图像高度
    
    Returns:
        (coco_bbox, area): COCO格式bbox和面积
    """
    x_center, y_center, w, h = yolo_bbox
    
    # 转换为像素坐标
    x_center_px = x_center * img_width
    y_center_px = y_center * img_height
    w_px = w * img_width
    h_px = h * img_height
    
    # 转换为左上角坐标
    x_min = x_center_px - w_px / 2
    y_min = y_center_px - h_px / 2
    
    # 确保不越界
    x_min = max(0, x_min)
    y_min = max(0, y_min)
    w_px = min(w_px, img_width - x_min)
    h_px = min(h_px, img_height - y_min)
    
    area = w_px * h_px
    
    return [x_min, y_min, w_px, h_px], area


def parse_yolo_label(label_path: str) -> List[Tuple[int, List[float]]]:
    """
    解析YOLO格式的标注文件
    
    Args:
        label_path: 标注文件路径
    
    Returns:
        [(class_id, [x_center, y_center, width, height]), ...]
    """
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
