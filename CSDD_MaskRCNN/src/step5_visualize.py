#!/usr/bin/env python3
"""
第5步：结果可视化
可视化检测和分割结果

功能:
- 检测结果可视化（bbox + 类别 + 置信度）
- 分割结果可视化（mask叠加）
- GT vs Prediction对比
- 保存可视化图片
"""

import os
import sys
import cv2
import json
import argparse
import numpy as np
from tqdm import tqdm
import random

# 添加项目路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.visualization import (
    draw_bbox,
    draw_mask,
    visualize_instance_results,
    visualize_gt_vs_pred,
    save_visualization,
    DEFAULT_COLORS,
    CLASS_NAMES
)


def parse_args():
    parser = argparse.ArgumentParser(description='可视化结果')
    parser.add_argument(
        '--config',
        type=str,
        default='configs/mask_rcnn_r101_fpn_csdd_4gpu.py',
        help='配置文件路径'
    )
    parser.add_argument(
        '--checkpoint',
        type=str,
        required=True,
        help='模型权重路径'
    )
    parser.add_argument(
        '--test-dir',
        type=str,
        default='/root/autodl-tmp/CSDD_coco/test',
        help='测试图像目录'
    )
    parser.add_argument(
        '--test-ann',
        type=str,
        default='data/CSDD_coco/annotations/instances_test.json',
        help='测试集标注文件'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='outputs/visualizations',
        help='可视化输出目录'
    )
    parser.add_argument(
        '--score-thr',
        type=float,
        default=0.5,
        help='置信度阈值'
    )
    parser.add_argument(
        '--num-images',
        type=int,
        default=50,
        help='可视化图像数量'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda:0',
        help='推理设备'
    )
    parser.add_argument(
        '--random-select',
        action='store_true',
        help='随机选择图像'
    )
    return parser.parse_args()


def load_gt_annotations(ann_path):
    """加载GT标注"""
    with open(ann_path, 'r') as f:
        coco = json.load(f)
    
    # 按图像ID组织
    gt_by_image = {}
    for img in coco['images']:
        gt_by_image[img['id']] = {
            'file_name': img['file_name'],
            'width': img['width'],
            'height': img['height'],
            'annotations': []
        }
    
    for ann in coco['annotations']:
        img_id = ann['image_id']
        if img_id in gt_by_image:
            gt_by_image[img_id]['annotations'].append(ann)
    
    return gt_by_image, coco['categories']


def visualize_single_image(model, image_path, gt_anns, output_dir, score_thr=0.5):
    """可视化单张图像"""
    from mmdet.apis import inference_detector
    from pycocotools import mask as maskUtils
    
    # 读取图像
    image = cv2.imread(image_path)
    if image is None:
        return None
    
    h, w = image.shape[:2]
    
    # 推理
    result = inference_detector(model, image_path)
    pred_instances = result.pred_instances
    
    # 解析预测结果
    predictions = []
    if len(pred_instances) > 0:
        bboxes = pred_instances.bboxes.cpu().numpy()
        scores = pred_instances.scores.cpu().numpy()
        labels = pred_instances.labels.cpu().numpy()
        
        if hasattr(pred_instances, 'masks'):
            masks = pred_instances.masks.cpu().numpy()
        else:
            masks = [None] * len(bboxes)
        
        for i in range(len(bboxes)):
            if scores[i] < score_thr:
                continue
            
            x1, y1, x2, y2 = bboxes[i]
            bbox = [float(x1), float(y1), float(x2-x1), float(y2-y1)]
            
            pred = {
                'bbox': bbox,
                'category_id': int(labels[i]) + 1,
                'score': float(scores[i])
            }
            
            if masks[i] is not None:
                pred['mask'] = masks[i].astype(np.uint8)
            
            predictions.append(pred)
    
    # 解析GT
    gt_list = []
    for ann in gt_anns:
        gt_item = {
            'bbox': ann['bbox'],
            'category_id': ann['category_id']
        }
        
        # 解析segmentation
        if 'segmentation' in ann:
            seg = ann['segmentation']
            if isinstance(seg, list):
                # polygon格式
                mask = np.zeros((h, w), dtype=np.uint8)
                for poly in seg:
                    pts = np.array(poly).reshape(-1, 2).astype(np.int32)
                    cv2.fillPoly(mask, [pts], 1)
                gt_item['mask'] = mask
            elif isinstance(seg, dict):
                # RLE格式
                mask = maskUtils.decode(seg)
                gt_item['mask'] = mask
        
        gt_list.append(gt_item)
    
    # 生成可视化图像
    results = {}
    
    # 1. 检测结果
    det_img = image.copy()
    for pred in predictions:
        det_img = draw_bbox(det_img, pred['bbox'], pred['category_id'], pred['score'])
    results['detection'] = det_img
    
    # 2. 分割结果
    seg_img = image.copy()
    for pred in predictions:
        if 'mask' in pred:
            seg_img = draw_mask(seg_img, pred['mask'], pred['category_id'], alpha=0.5)
    results['segmentation'] = seg_img
    
    # 3. 实例分割（bbox + mask）
    inst_img = visualize_instance_results(image, predictions, score_thr)
    results['instance'] = inst_img
    
    # 4. GT vs Pred对比
    comparison = visualize_gt_vs_pred(image, gt_list, predictions, score_thr)
    results['comparison'] = comparison
    
    return results


def main():
    args = parse_args()
    
    print("=" * 60)
    print("CSDD 结果可视化")
    print("=" * 60)
    
    # 检查文件
    if not os.path.exists(args.config):
        print(f"[ERROR] 配置文件不存在: {args.config}")
        return
    
    if not os.path.exists(args.checkpoint):
        print(f"[ERROR] 模型权重不存在: {args.checkpoint}")
        return
    
    # 导入MMDetection
    try:
        from mmengine.config import Config
        from mmdet.apis import init_detector
    except ImportError as e:
        print(f"[ERROR] 请先安装MMDetection: {e}")
        return
    
    # 加载模型
    print("加载模型...")
    cfg = Config.fromfile(args.config)
    model = init_detector(cfg, args.checkpoint, device=args.device)
    
    # 加载GT
    print("加载标注...")
    gt_by_image, categories = load_gt_annotations(args.test_ann)
    
    # 更新类别名称
    for cat in categories:
        CLASS_NAMES[cat['id']] = cat['name']
    
    # 获取图像列表
    image_ids = list(gt_by_image.keys())
    
    if args.random_select:
        random.shuffle(image_ids)
    
    image_ids = image_ids[:args.num_images]
    
    print(f"将可视化 {len(image_ids)} 张图像")
    
    # 创建输出目录
    det_dir = os.path.join(args.output_dir, 'detection')
    seg_dir = os.path.join(args.output_dir, 'segmentation')
    inst_dir = os.path.join(args.output_dir, 'instance')
    comp_dir = os.path.join(args.output_dir, 'comparison')
    
    os.makedirs(det_dir, exist_ok=True)
    os.makedirs(seg_dir, exist_ok=True)
    os.makedirs(inst_dir, exist_ok=True)
    os.makedirs(comp_dir, exist_ok=True)
    
    # 可视化
    for img_id in tqdm(image_ids, desc="可视化"):
        img_info = gt_by_image[img_id]
        image_path = os.path.join(args.test_dir, img_info['file_name'])
        
        if not os.path.exists(image_path):
            continue
        
        results = visualize_single_image(
            model, image_path, img_info['annotations'],
            args.output_dir, args.score_thr
        )
        
        if results is None:
            continue
        
        # 保存
        base_name = os.path.splitext(img_info['file_name'])[0]
        
        save_visualization(results['detection'], os.path.join(det_dir, f'{base_name}_det.jpg'))
        save_visualization(results['segmentation'], os.path.join(seg_dir, f'{base_name}_seg.jpg'))
        save_visualization(results['instance'], os.path.join(inst_dir, f'{base_name}_inst.jpg'))
        save_visualization(results['comparison'], os.path.join(comp_dir, f'{base_name}_comp.jpg'))
    
    print("\n" + "=" * 60)
    print("可视化完成!")
    print("=" * 60)
    print(f"检测结果: {det_dir}")
    print(f"分割结果: {seg_dir}")
    print(f"实例分割: {inst_dir}")
    print(f"GT对比: {comp_dir}")


if __name__ == '__main__':
    main()
