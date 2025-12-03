#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Script para evaluar un modelo entrenado y calcular F1 score (Dice) e IoU
Soporta tanto LaneNet como LaneNetPlus
"""

import time
import os
import sys
import argparse

import torch
from dataloader.data_loaders import TusimpleSet
from dataloader.transformers import Rescale
from model.lanenet.LaneNet import LaneNet
from model.lanenet.LaneNetPlus import LaneNetPlus
from torch.utils.data import DataLoader
from torchvision import transforms
from model.eval_function import Eval_Score

import numpy as np
import pandas as pd
import cv2

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate trained model and calculate F1 score and IoU')
    parser.add_argument("--model", type=str, required=True, help="Path to trained model (.pth file)")
    parser.add_argument("--dataset", type=str, required=True, help="Path to dataset directory containing train.txt/val.txt/test.txt")
    parser.add_argument("--split", type=str, default="val", choices=["train", "val", "test"], 
                       help="Which split to evaluate (default: val)")
    parser.add_argument("--model_type", type=str, default="ENet", choices=["ENet", "UNet", "DeepLabv3+"],
                       help="Backbone architecture (default: ENet)")
    parser.add_argument("--use_lanenet_plus", action="store_true", 
                       help="Use LaneNetPlus instead of LaneNet")
    parser.add_argument("--use_attention", action="store_true",
                       help="Use self-attention (only for LaneNetPlus)")
    parser.add_argument("--use_multitask", action="store_true",
                       help="Use multi-task learning (only for LaneNetPlus)")
    parser.add_argument("--height", type=int, default=256, help="Resize height (default: 256)")
    parser.add_argument("--width", type=int, default=512, help="Resize width (default: 512)")
    parser.add_argument("--bs", type=int, default=1, help="Batch size (default: 1)")
    parser.add_argument("--save_csv", type=str, default=None, 
                       help="Path to save detailed results CSV (optional)")
    parser.add_argument("--verbose", action="store_true", help="Print per-image results")
    
    return parser.parse_args()


def evaluation():
    args = parse_args()
    
    resize_height = args.height
    resize_width = args.width

    # Data transforms
    data_transform = transforms.Compose([
        transforms.Resize((resize_height, resize_width)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    target_transforms = transforms.Compose([
        Rescale((resize_width, resize_height)),
    ])

    # Load dataset
    dataset_file = os.path.join(args.dataset, f'{args.split}.txt')
    
    if not os.path.exists(dataset_file):
        print(f"Error: Dataset file not found: {dataset_file}")
        print(f"Available files in {args.dataset}:")
        for f in os.listdir(args.dataset):
            print(f"  - {f}")
        sys.exit(1)
    
    print(f"Loading dataset from: {dataset_file}")
    Eval_Dataset = TusimpleSet(
        dataset_file, 
        transform=data_transform, 
        target_transform=target_transforms,
        drivable_dir=None,  # Not needed for evaluation
        use_drivable=False
    )
    eval_dataloader = DataLoader(Eval_Dataset, batch_size=args.bs, shuffle=False)
    
    print(f"Dataset loaded: {len(Eval_Dataset)} images")

    # Load model
    print(f"Loading model from: {args.model}")
    model_path = args.model
    
    if args.use_lanenet_plus:
        model = LaneNetPlus(
            arch=args.model_type,
            use_attention=args.use_attention,
            use_multitask=args.use_multitask,
            freeze_encoder=False
        )
        print(f"Model: LaneNetPlus")
        print(f"  - Architecture: {args.model_type}")
        print(f"  - Attention: {args.use_attention}")
        print(f"  - Multi-task: {args.use_multitask}")
    else:
        model = LaneNet(arch=args.model_type, freeze_encoder=False)
        print(f"Model: LaneNet")
        print(f"  - Architecture: {args.model_type}")
    
    # Load weights
    state_dict = torch.load(model_path, map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.eval()
    model.to(DEVICE)
    
    print(f"Model loaded successfully!")
    print("-" * 80)

    # Evaluate
    print(f"\nEvaluating on {args.split} set...")
    print("-" * 80)
    
    results = []
    iou_sum, dice_sum = 0, 0
    total_images = 0
    
    start_time = time.time()
    
    with torch.no_grad():
        for batch_idx, (x, target, _) in enumerate(eval_dataloader):
            y = model(x.to(DEVICE))
            
            # Get binary segmentation prediction
            # Handle both LaneNet and LaneNetPlus outputs
            if 'lane_pred' in y:
                # LaneNetPlus: lane_pred has shape [batch, 1, H, W] or [batch, H, W]
                binary_pred = y['lane_pred'].to('cpu')
            elif 'binary_seg_pred' in y:
                # LaneNet or LaneNetPlus compatibility: binary_seg_pred
                binary_pred = y['binary_seg_pred'].to('cpu')
            else:
                raise ValueError("No binary/lane prediction found in model output")
            
            # Squeeze to remove batch and channel dimensions if needed
            # Expected final shape: [batch, H, W] or [H, W] for single image
            y_pred = torch.squeeze(binary_pred).numpy()
            
            # Handle batch dimension
            if len(y_pred.shape) == 2:  # Single image: [H, W]
                y_pred_batch = [y_pred]
            else:  # Batch: [batch, H, W]
                y_pred_batch = y_pred
            
            # Convert target to numpy (target is already binary mask from dataloader)
            y_true = target.numpy()
            
            # Process each image in the batch
            batch_size = len(y_pred_batch)
            for img_idx in range(batch_size):
                y_pred_img = y_pred_batch[img_idx]  # Shape: [H, W]
                y_true_img = y_true[img_idx]  # Shape: [H, W, C] or [H, W]
                
                # Convert prediction to binary (0-1 range)
                # LaneNetPlus returns already thresholded, but ensure it's in [0, 1] range
                y_pred_binary = (y_pred_img > 0).astype(np.float32)
                
                # Handle target shape - it might be RGB or grayscale
                if len(y_true_img.shape) == 3:
                    # Convert RGB mask to binary (any non-zero pixel)
                    y_true_binary = (np.sum(y_true_img, axis=2) > 0).astype(np.float32)
                else:
                    y_true_binary = (y_true_img > 0).astype(np.float32)
                
                # Resize predictions to match target if needed
                if y_pred_binary.shape != y_true_binary.shape:
                    y_pred_binary = cv2.resize(y_pred_binary, 
                                             (y_true_binary.shape[1], y_true_binary.shape[0]),
                                             interpolation=cv2.INTER_NEAREST)
                
                # Calculate metrics
                Score = Eval_Score(y_pred_binary, y_true_binary)
                dice = Score.Dice()
                iou = Score.IoU()
                
                # Store result
                image_path = Eval_Dataset._gt_img_list[batch_idx * args.bs + img_idx]
                results.append({
                    'image_path': image_path,
                    'iou': float(iou),
                    'dice': float(dice)
                })
                
                iou_sum += iou
                dice_sum += dice
                total_images += 1
                
                if args.verbose:
                    img_name = os.path.basename(image_path)
                    print(f"  [{total_images}/{len(Eval_Dataset)}] {img_name}: IoU={iou:.4f}, F1={dice:.4f}")
    
    elapsed_time = time.time() - start_time
    
    # Calculate averages
    avg_iou = iou_sum / total_images if total_images > 0 else 0
    avg_dice = dice_sum / total_images if total_images > 0 else 0
    
    # Print results
    print("\n" + "=" * 80)
    print("EVALUATION RESULTS")
    print("=" * 80)
    print(f"Dataset: {args.split} set")
    print(f"Total images evaluated: {total_images}")
    print(f"Evaluation time: {elapsed_time:.2f} seconds ({elapsed_time/total_images:.3f} sec/image)")
    print("-" * 80)
    print(f"Average IoU:  {avg_iou:.4f}")
    print(f"Average F1 (Dice):  {avg_dice:.4f}")
    print("=" * 80)
    
    # Save CSV if requested
    if args.save_csv:
        df = pd.DataFrame(results)
        df.to_csv(args.save_csv, index=False, encoding='utf-8')
        print(f"\nDetailed results saved to: {args.save_csv}")
        
        # Print statistics
        print(f"\nStatistics:")
        print(f"  IoU - Min: {df['iou'].min():.4f}, Max: {df['iou'].max():.4f}, Std: {df['iou'].std():.4f}")
        print(f"  F1  - Min: {df['dice'].min():.4f}, Max: {df['dice'].max():.4f}, Std: {df['dice'].std():.4f}")
    
    return avg_iou, avg_dice


if __name__ == "__main__":
    evaluation()

