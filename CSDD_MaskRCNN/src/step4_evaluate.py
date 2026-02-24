#!/usr/bin/env python3
"""
第4步：模型评估
在测试集上评估模型性能，计算mAP和mIoU

功能:
- 加载训练好的模型
- 在测试集上进行推理
- 计算检测指标 (mAP, AP@0.5, AP@0.75, 各类别AP)
- 计算分割指标 (mIoU, 各类别IoU)
- 生成评估报告
"""

import os
import sys
import json
import argparse
import numpy as np
from datetime import datetime
from collections import defaultdict
from tqdm import tqdm

# 添加项目路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def parse_args():
    parser = argparse.ArgumentParser(description='评估Mask R-CNN')
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
        '--test-ann',
        type=str,
        default='/root/autodl-tmp/CSDD_coco/annotations/instances_test.json',
        help='测试集标注文件'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='outputs/eval_results',
        help='评估结果输出目录'
    )
    parser.add_argument(
        '--score-thr',
        type=float,
        default=0.3,
        help='置信度阈值'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda:0',
        help='推理设备'
    )
    return parser.parse_args()


def evaluate_with_mmdet(config_path, checkpoint_path, test_ann_path, output_dir, device='cuda:0'):
    """
    使用MMDetection进行评估
    """
    from mmengine.config import Config
    from mmengine.runner import Runner
    from mmdet.apis import init_detector, inference_detector
    from mmdet.evaluation import CocoMetric
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    from pycocotools import mask as maskUtils
    
    print("加载配置和模型...")
    cfg = Config.fromfile(config_path)
    
    # 修改测试配置
    cfg.test_dataloader.dataset.ann_file = test_ann_path
    #cfg.test_evaluator.ann_file = test_ann_path
    if isinstance(cfg.test_evaluator, list):
        for evaluator in cfg.test_evaluator:
            evaluator['ann_file'] = test_ann_path
    else:
        cfg.test_evaluator.ann_file = test_ann_path

    # 初始化模型
    model = init_detector(cfg, checkpoint_path, device=device)
    
    # 加载GT
    coco_gt = COCO(test_ann_path)
    img_ids = coco_gt.getImgIds()
    
    print(f"测试图像数量: {len(img_ids)}")
    
    # 推理
    results_bbox = []
    results_segm = []
    
    data_root = cfg.test_dataloader.dataset.data_root
    img_prefix = cfg.test_dataloader.dataset.data_prefix.get('img', '')
    
    for img_id in tqdm(img_ids, desc="推理中"):
        img_info = coco_gt.loadImgs(img_id)[0]
        img_path = os.path.join(data_root, img_prefix, img_info['file_name'])
        
        # 推理
        result = inference_detector(model, img_path)
        
        # 解析结果
        pred_instances = result.pred_instances
        
        if len(pred_instances) == 0:
            continue
        
        bboxes = pred_instances.bboxes.cpu().numpy()
        scores = pred_instances.scores.cpu().numpy()
        labels = pred_instances.labels.cpu().numpy()
        
        # 处理mask
        if hasattr(pred_instances, 'masks'):
            masks = pred_instances.masks.cpu().numpy()
        else:
            masks = None
        
        for i in range(len(bboxes)):
            x1, y1, x2, y2 = bboxes[i]
            bbox = [float(x1), float(y1), float(x2-x1), float(y2-y1)]
            
            # bbox结果
            results_bbox.append({
                'image_id': img_id,
                'category_id': int(labels[i]) + 1,  # 转为1-indexed
                'bbox': bbox,
                'score': float(scores[i])
            })
            
            # mask结果
            if masks is not None:
                mask = masks[i]
                rle = maskUtils.encode(np.asfortranarray(mask.astype(np.uint8)))
                rle['counts'] = rle['counts'].decode('utf-8')
                
                results_segm.append({
                    'image_id': img_id,
                    'category_id': int(labels[i]) + 1,
                    'segmentation': rle,
                    'score': float(scores[i])
                })
    
    # 保存预测结果
    os.makedirs(output_dir, exist_ok=True)
    
    bbox_results_path = os.path.join(output_dir, 'bbox_predictions.json')
    with open(bbox_results_path, 'w') as f:
        json.dump(results_bbox, f)
    
    segm_results_path = os.path.join(output_dir, 'segm_predictions.json')
    with open(segm_results_path, 'w') as f:
        json.dump(results_segm, f)
    
    print(f"\n预测结果已保存")
    
    # COCO评估 - 检测
    print("\n" + "=" * 60)
    print("检测评估 (BBox)")
    print("=" * 60)
    
    coco_dt_bbox = coco_gt.loadRes(bbox_results_path)
    coco_eval_bbox = COCOeval(coco_gt, coco_dt_bbox, 'bbox')
    coco_eval_bbox.evaluate()
    coco_eval_bbox.accumulate()
    coco_eval_bbox.summarize()
    
    bbox_metrics = {
        'mAP': coco_eval_bbox.stats[0],
        'AP@0.5': coco_eval_bbox.stats[1],
        'AP@0.75': coco_eval_bbox.stats[2],
        'AP_small': coco_eval_bbox.stats[3],
        'AP_medium': coco_eval_bbox.stats[4],
        'AP_large': coco_eval_bbox.stats[5],
    }
    
    # COCO评估 - 分割
    print("\n" + "=" * 60)
    print("分割评估 (Segmentation)")
    print("=" * 60)
    
    coco_dt_segm = coco_gt.loadRes(segm_results_path)
    coco_eval_segm = COCOeval(coco_gt, coco_dt_segm, 'segm')
    coco_eval_segm.evaluate()
    coco_eval_segm.accumulate()
    coco_eval_segm.summarize()
    
    segm_metrics = {
        'mAP': coco_eval_segm.stats[0],
        'AP@0.5': coco_eval_segm.stats[1],
        'AP@0.75': coco_eval_segm.stats[2],
        'AP_small': coco_eval_segm.stats[3],
        'AP_medium': coco_eval_segm.stats[4],
        'AP_large': coco_eval_segm.stats[5],
    }
    
    # 计算每类别指标
    print("\n" + "=" * 60)
    print("各类别详细指标")
    print("=" * 60)
    
    class_names = {cat['id']: cat['name'] for cat in coco_gt.dataset['categories']}
    per_class_metrics = {}
    
    for cat_id, cat_name in class_names.items():
        # bbox
        coco_eval_bbox.params.catIds = [cat_id]
        coco_eval_bbox.evaluate()
        coco_eval_bbox.accumulate()
        
        # segm
        coco_eval_segm.params.catIds = [cat_id]
        coco_eval_segm.evaluate()
        coco_eval_segm.accumulate()
        
        per_class_metrics[cat_name] = {
            'bbox_AP': float(coco_eval_bbox.stats[0]),
            'bbox_AP50': float(coco_eval_bbox.stats[1]),
            'bbox_AP75': float(coco_eval_bbox.stats[2]),
            'segm_AP': float(coco_eval_segm.stats[0]),
            'segm_AP50': float(coco_eval_segm.stats[1]),
            'segm_AP75': float(coco_eval_segm.stats[2]),
        }
        
        print(f"\n{cat_name}:")
        print(f"  BBox - AP: {per_class_metrics[cat_name]['bbox_AP']*100:.2f}%, "
              f"AP@0.5: {per_class_metrics[cat_name]['bbox_AP50']*100:.2f}%, "
              f"AP@0.75: {per_class_metrics[cat_name]['bbox_AP75']*100:.2f}%")
        print(f"  Segm - AP: {per_class_metrics[cat_name]['segm_AP']*100:.2f}%, "
              f"AP@0.5: {per_class_metrics[cat_name]['segm_AP50']*100:.2f}%, "
              f"AP@0.75: {per_class_metrics[cat_name]['segm_AP75']*100:.2f}%")
    
    # 计算mIoU (基于分割AP)
    segm_aps = [v['segm_AP'] for v in per_class_metrics.values()]
    mIoU_approx = np.mean(segm_aps) if segm_aps else 0
    
    # 汇总结果
    all_metrics = {
        'detection': bbox_metrics,
        'segmentation': segm_metrics,
        'per_class': per_class_metrics,
        'mIoU_approx': mIoU_approx,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    # 保存结果
    metrics_path = os.path.join(output_dir, 'evaluation_metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(all_metrics, f, indent=2)
    
    # 生成报告
    generate_report(all_metrics, per_class_metrics, output_dir)
    
    return all_metrics


def generate_report(metrics, per_class_metrics, output_dir):
    """生成Markdown格式的评估报告"""
    
    report = f"""# CSDD 缺陷检测与分割评估报告

生成时间: {metrics['timestamp']}

## 1. 检测性能 (Detection - BBox)

| 指标 | 数值 |
|------|------|
| mAP@[.5:.95] | {metrics['detection']['mAP']*100:.2f}% |
| AP@0.5 | {metrics['detection']['AP@0.5']*100:.2f}% |
| AP@0.75 | {metrics['detection']['AP@0.75']*100:.2f}% |
| AP (small) | {metrics['detection']['AP_small']*100:.2f}% |
| AP (medium) | {metrics['detection']['AP_medium']*100:.2f}% |
| AP (large) | {metrics['detection']['AP_large']*100:.2f}% |

## 2. 分割性能 (Segmentation - Mask)

| 指标 | 数值 |
|------|------|
| mAP@[.5:.95] | {metrics['segmentation']['mAP']*100:.2f}% |
| AP@0.5 | {metrics['segmentation']['AP@0.5']*100:.2f}% |
| AP@0.75 | {metrics['segmentation']['AP@0.75']*100:.2f}% |
| AP (small) | {metrics['segmentation']['AP_small']*100:.2f}% |
| AP (medium) | {metrics['segmentation']['AP_medium']*100:.2f}% |
| AP (large) | {metrics['segmentation']['AP_large']*100:.2f}% |

## 3. 各类别详细性能

### 检测 (BBox)

| 类别 | AP@[.5:.95] | AP@0.5 | AP@0.75 |
|------|-------------|--------|---------|
"""
    
    for class_name, class_metrics in per_class_metrics.items():
        report += f"| {class_name} | {class_metrics['bbox_AP']*100:.2f}% | {class_metrics['bbox_AP50']*100:.2f}% | {class_metrics['bbox_AP75']*100:.2f}% |\n"
    
    report += f"""
### 分割 (Mask)

| 类别 | AP@[.5:.95] | AP@0.5 | AP@0.75 |
|------|-------------|--------|---------|
"""
    
    for class_name, class_metrics in per_class_metrics.items():
        report += f"| {class_name} | {class_metrics['segm_AP']*100:.2f}% | {class_metrics['segm_AP50']*100:.2f}% | {class_metrics['segm_AP75']*100:.2f}% |\n"
    
    report += f"""
## 4. 汇总

- **检测 mAP**: {metrics['detection']['mAP']*100:.2f}%
- **分割 mAP**: {metrics['segmentation']['mAP']*100:.2f}%
- **近似 mIoU**: {metrics['mIoU_approx']*100:.2f}%

## 5. 说明

- mAP@[.5:.95]: COCO标准mAP，IoU阈值从0.5到0.95，步长0.05
- AP@0.5: IoU阈值为0.5时的AP（VOC标准）
- AP@0.75: IoU阈值为0.75时的AP（严格标准）
- AP (small/medium/large): 按目标尺寸分类的AP
"""
    
    report_path = os.path.join(output_dir, 'evaluation_report.md')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"\n评估报告已保存: {report_path}")


def main():
    args = parse_args()
    
    print("=" * 60)
    print("CSDD Mask R-CNN 评估")
    print("=" * 60)
    print(f"配置文件: {args.config}")
    print(f"模型权重: {args.checkpoint}")
    print(f"测试集标注: {args.test_ann}")
    print(f"输出目录: {args.output_dir}")
    print("=" * 60)
    
    # 检查文件
    if not os.path.exists(args.config):
        print(f"[ERROR] 配置文件不存在: {args.config}")
        return
    
    if not os.path.exists(args.checkpoint):
        print(f"[ERROR] 模型权重不存在: {args.checkpoint}")
        return
    
    if not os.path.exists(args.test_ann):
        print(f"[ERROR] 测试集标注不存在: {args.test_ann}")
        return
    
    # 评估
    metrics = evaluate_with_mmdet(
        args.config,
        args.checkpoint,
        args.test_ann,
        args.output_dir,
        args.device
    )
    
    print("\n" + "=" * 60)
    print("评估完成!")
    print("=" * 60)


if __name__ == '__main__':
    main()
