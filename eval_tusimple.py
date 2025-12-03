"""
Evaluate lane detection on TUSIMPLE test set and find worst accuracy images
Reads labels from JSON format and computes IoU/F1 scores per image
"""
import os
import sys
import json
import argparse
from collections import defaultdict

import torch
from model.lanenet.LaneNet import LaneNet
from torchvision import transforms
import numpy as np
from PIL import Image
import cv2

from model.eval_function import Eval_Score

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def load_test_data(img_path, transform):
    img = Image.open(img_path)
    # Convert RGBA to RGB if necessary
    if img.mode == 'RGBA':
        rgb_img = Image.new('RGB', img.size, (255, 255, 255))
        rgb_img.paste(img, mask=img.split()[3])
        img = rgb_img
    elif img.mode == 'L':
        img = img.convert('RGB')
    elif img.mode != 'RGB':
        img = img.convert('RGB')
    img = transform(img)
    return img


def tusimple_lanes_to_binary_mask(lanes, h_samples, img_height, img_width):
    """
    Convert TUSIMPLE lane format (list of x coordinates per h_sample) to binary mask
    
    Args:
        lanes: List of lanes, each lane is a list of x coordinates (or -2 if not present)
        h_samples: List of y coordinates (heights) where lanes are sampled
        img_height: Height of the image
        img_width: Width of the image
    
    Returns:
        binary_mask: Binary mask (0 or 255) of lane markings
    """
    binary_mask = np.zeros((img_height, img_width), dtype=np.uint8)
    
    for lane in lanes:
        lane_points = []
        for idx, x_coord in enumerate(lane):
            if x_coord != -2 and 0 <= x_coord < img_width:
                y_coord = h_samples[idx]
                if 0 <= y_coord < img_height:
                    lane_points.append((int(x_coord), int(y_coord)))
        
        # Draw lane as polylines (thicker lines for better matching)
        if len(lane_points) >= 2:
            lane_pts = np.array(lane_points, np.int32)
            cv2.polylines(binary_mask, [lane_pts], isClosed=False, color=255, thickness=5)
    
    return binary_mask


def evaluate_tusimple_test_set(test_json_path, dataset_base_dir, model_path, 
                                model_type='ENet', resize_width=512, resize_height=256,
                                save_worst_n=20, output_dir='worst_accuracy_results'):
    """
    Evaluate all images in TUSIMPLE test set and find worst accuracy images
    
    Args:
        test_json_path: Path to test_label.json file
        dataset_base_dir: Base directory of TUSIMPLE dataset
        model_path: Path to model weights
        model_type: Model architecture type
        resize_width: Width to resize images
        resize_height: Height to resize images
        save_worst_n: Number of worst accuracy images to save
        output_dir: Directory to save worst accuracy results
    """
    
    # Load model
    print(f"Loading model from {model_path}...")
    model = LaneNet(arch=model_type)
    state_dict = torch.load(model_path, map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.eval()
    model.to(DEVICE)
    print("Model loaded successfully\n")
    
    # Setup transforms
    data_transform = transforms.Compose([
        transforms.Resize((resize_height, resize_width)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Read test labels from JSON
    print(f"Reading labels from {test_json_path}...")
    test_entries = []
    with open(test_json_path, 'r') as f:
        for line in f:
            if line.strip():
                test_entries.append(json.loads(line))
    
    print(f"Found {len(test_entries)} test images\n")
    
    # Store results for each image
    results = []
    
    print("Evaluating images...")
    print("="*80)
    
    # Process each test entry
    for idx, entry in enumerate(test_entries):
        raw_file = entry['raw_file']
        lanes = entry['lanes']
        h_samples = entry['h_samples']
        
        # Construct full image path
        img_path = os.path.join(dataset_base_dir, raw_file)
        
        if not os.path.exists(img_path):
            print(f"Warning: Image not found: {img_path}")
            continue
        
        print(f"Processing {idx+1}/{len(test_entries)}: {raw_file}")
        
        # Load and preprocess image
        try:
            dummy_input = load_test_data(img_path, data_transform).to(DEVICE)
            dummy_input = torch.unsqueeze(dummy_input, dim=0)
        except Exception as e:
            print(f"  Error loading image: {e}")
            continue
        
        # Run model
        with torch.no_grad():
            outputs = model(dummy_input)
        
        # Get predictions
        y_pred = torch.squeeze(outputs['binary_seg_pred']).detach().to('cpu').numpy()
        
        # Load original image to get dimensions
        original_img = cv2.imread(img_path)
        if original_img is None:
            print(f"  Error: Could not load original image")
            continue
        
        orig_height, orig_width = original_img.shape[:2]
        
        # Resize prediction to original image size
        if y_pred.shape != (orig_height, orig_width):
            y_pred_resized = cv2.resize(y_pred.astype(np.uint8), 
                                       (orig_width, orig_height), 
                                       interpolation=cv2.INTER_NEAREST)
        else:
            y_pred_resized = y_pred
        
        # Convert TUSIMPLE labels to binary mask
        y_true = tusimple_lanes_to_binary_mask(lanes, h_samples, orig_height, orig_width)
        
        # Resize ground truth to match prediction size if needed
        pred_height, pred_width = y_pred_resized.shape
        if y_true.shape != (pred_height, pred_width):
            y_true_resized = cv2.resize(y_true, (pred_width, pred_height), 
                                       interpolation=cv2.INTER_NEAREST)
            y_true_resized = (y_true_resized > 127).astype(np.uint8)
        else:
            y_true_resized = (y_true > 127).astype(np.uint8)
        
        # Calculate metrics
        Score = Eval_Score(y_pred_resized, y_true_resized)
        iou = Score.IoU()
        dice = Score.Dice()
        
        # Store result
        results.append({
            'image_path': img_path,
            'raw_file': raw_file,
            'iou': float(iou),
            'dice': float(dice),
            'pred_mask': y_pred_resized,
            'gt_mask': y_true_resized,
            'original_img': original_img
        })
        
        print(f"  IoU: {iou:.4f}, F1 (Dice): {dice:.4f}")
    
    print("\n" + "="*80)
    print(f"\nEvaluation complete! Processed {len(results)} images")
    
    if len(results) == 0:
        print("No images were successfully processed.")
        return
    
    # Calculate average metrics
    avg_iou = np.mean([r['iou'] for r in results])
    avg_dice = np.mean([r['dice'] for r in results])
    
    print(f"\nAverage IoU: {avg_iou:.4f}")
    print(f"Average F1 (Dice): {avg_dice:.4f}")
    
    # Sort by worst accuracy (lowest IoU)
    results_sorted = sorted(results, key=lambda x: x['iou'])
    
    # Print worst N images
    print(f"\n{'='*80}")
    print(f"WORST {save_worst_n} IMAGES (by IoU):")
    print(f"{'='*80}")
    for i, result in enumerate(results_sorted[:save_worst_n]):
        print(f"{i+1}. {result['raw_file']}: IoU={result['iou']:.4f}, F1={result['dice']:.4f}")
    
    # Save worst N images with visualizations
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\nSaving worst {save_worst_n} images to {output_dir}/...")
    for i, result in enumerate(results_sorted[:save_worst_n]):
        # Create visualization: original + GT overlay + Pred overlay
        orig_img = result['original_img'].copy()
        gt_mask = result['gt_mask']
        pred_mask = result['pred_mask']
        
        # Create overlay images
        gt_overlay = orig_img.copy()
        gt_overlay[gt_mask > 0] = [0, 255, 0]  # Green for GT
        gt_overlay_final = cv2.addWeighted(orig_img, 0.6, gt_overlay, 0.4, 0)
        
        pred_overlay = orig_img.copy()
        pred_mask_binary = (pred_mask > 0).astype(np.uint8)
        pred_overlay[pred_mask_binary > 0] = [0, 0, 255]  # Red for prediction
        pred_overlay_final = cv2.addWeighted(orig_img, 0.6, pred_overlay, 0.4, 0)
        
        # Combine all three side by side
        h, w = orig_img.shape[:2]
        combined = np.hstack([orig_img, gt_overlay_final, pred_overlay_final])
        
        # Add text with metrics
        text = f"IoU: {result['iou']:.4f} | F1: {result['dice']:.4f}"
        cv2.putText(combined, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Save
        filename = os.path.basename(result['raw_file']).replace('/', '_')
        output_path = os.path.join(output_dir, f"worst_{i+1:03d}_{filename}")
        cv2.imwrite(output_path, combined)
        
        # Also save individual masks
        cv2.imwrite(output_path.replace('.jpg', '_gt_mask.jpg'), gt_mask * 255)
        cv2.imwrite(output_path.replace('.jpg', '_pred_mask.jpg'), pred_mask_binary * 255)
    
    # Save summary CSV
    import pandas as pd
    summary_data = []
    for i, result in enumerate(results_sorted):
        summary_data.append({
            'rank': i+1,
            'raw_file': result['raw_file'],
            'iou': result['iou'],
            'f1_dice': result['dice']
        })
    
    df = pd.DataFrame(summary_data)
    csv_path = os.path.join(output_dir, 'accuracy_results.csv')
    df.to_csv(csv_path, index=False)
    print(f"\nSaved summary to {csv_path}")
    
    print(f"\n{'='*80}")
    print(f"Results saved in: {output_dir}/")
    print(f"Worst image: {results_sorted[0]['raw_file']} (IoU: {results_sorted[0]['iou']:.4f})")
    print(f"{'='*80}")


def main():
    parser = argparse.ArgumentParser(description='Evaluate TUSIMPLE test set and find worst accuracy images')
    parser.add_argument("--test_json", type=str, required=True,
                       help="Path to test_label.json file")
    parser.add_argument("--dataset_dir", type=str, required=True,
                       help="Base directory of TUSIMPLE dataset (contains clips/)")
    parser.add_argument("--model", type=str, default='./log/best_model.pth',
                       help="Model path")
    parser.add_argument("--model_type", type=str, default='ENet', help="Model type")
    parser.add_argument("--width", type=int, default=512, help="Resize width")
    parser.add_argument("--height", type=int, default=256, help="Resize height")
    parser.add_argument("--worst_n", type=int, default=20,
                       help="Number of worst accuracy images to save")
    parser.add_argument("--output", type=str, default='./worst_accuracy_results',
                       help="Output directory for worst accuracy images")
    
    args = parser.parse_args()
    
    evaluate_tusimple_test_set(
        test_json_path=args.test_json,
        dataset_base_dir=args.dataset_dir,
        model_path=args.model,
        model_type=args.model_type,
        resize_width=args.width,
        resize_height=args.height,
        save_worst_n=args.worst_n,
        output_dir=args.output
    )


if __name__ == "__main__":
    main()

