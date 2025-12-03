"""
Evaluate lane detection on CULane dataset and find worst/best accuracy images
Reads labels from .lines.txt format and computes IoU/F1 scores per image
"""
import os
import sys
import argparse
import glob

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


def culane_lines_to_binary_mask(lines_file_path, img_height, img_width):
    """
    Convert CULane .lines.txt format to binary mask
    
    CULane format: Each line contains pairs of (x, y) coordinates for one lane
    Example: "x1 y1 x2 y2 x3 y3 ..."
    
    Args:
        lines_file_path: Path to .lines.txt file
        img_height: Height of the image
        img_width: Width of the image
    
    Returns:
        binary_mask: Binary mask (0 or 255) of lane markings
    """
    binary_mask = np.zeros((img_height, img_width), dtype=np.uint8)
    
    if not os.path.exists(lines_file_path):
        return binary_mask
    
    with open(lines_file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            # Parse coordinates: pairs of (x, y)
            coords = line.split()
            if len(coords) < 4:  # Need at least 2 points (4 coordinates)
                continue
            
            # Convert to list of points
            points = []
            for i in range(0, len(coords) - 1, 2):
                try:
                    x = float(coords[i])
                    y = float(coords[i + 1])
                    
                    # Filter out invalid coordinates (negative x means lane not visible)
                    if x >= 0 and 0 <= x < img_width and 0 <= y < img_height:
                        points.append((int(x), int(y)))
                except (ValueError, IndexError):
                    continue
            
            # Draw lane as polylines (thicker lines for better matching)
            if len(points) >= 2:
                lane_pts = np.array(points, np.int32)
                cv2.polylines(binary_mask, [lane_pts], isClosed=False, color=255, thickness=5)
    
    return binary_mask


def evaluate_culane_dataset(dataset_dir, model_path, 
                            model_type='ENet', resize_width=512, resize_height=256,
                            save_worst_n=10, save_best_n=10, output_dir='culane_eval_results'):
    """
    Evaluate all images in CULane dataset and find worst/best accuracy images
    
    Args:
        dataset_dir: Directory containing images and .lines.txt files
        model_path: Path to model weights
        model_type: Model architecture type
        resize_width: Width to resize images
        resize_height: Height to resize images
        save_worst_n: Number of worst accuracy images to save
        save_best_n: Number of best accuracy images to save
        output_dir: Directory to save results
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
    
    # Find all image files
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']
    image_paths = []
    for ext in image_extensions:
        image_paths.extend(glob.glob(os.path.join(dataset_dir, ext)))
    
    image_paths = sorted(image_paths)
    
    if not image_paths:
        print(f"No images found in {dataset_dir}")
        return
    
    print(f"Found {len(image_paths)} images to evaluate\n")
    
    # Store results for each image
    results = []
    
    print("Evaluating images...")
    print("="*80)
    
    # Process each image
    for idx, img_path in enumerate(image_paths):
        img_filename = os.path.basename(img_path)
        img_basename = os.path.splitext(img_filename)[0]
        
        # Find corresponding .lines.txt file
        lines_file_path = os.path.join(dataset_dir, f"{img_basename}.lines.txt")
        
        if not os.path.exists(lines_file_path):
            print(f"Warning: No label file found for {img_filename}, skipping...")
            continue
        
        print(f"Processing {idx+1}/{len(image_paths)}: {img_filename}")
        
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
        
        # Convert CULane labels to binary mask
        y_true = culane_lines_to_binary_mask(lines_file_path, orig_height, orig_width)
        
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
            'image_filename': img_filename,
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
    
    print(f"\n{'='*80}")
    print(f"ACCURACY GENERAL:")
    print(f"{'='*80}")
    print(f"Average IoU: {avg_iou:.4f}")
    print(f"Average F1 (Dice): {avg_dice:.4f}")
    print(f"Total images evaluated: {len(results)}")
    
    # Sort by worst and best accuracy
    results_sorted_worst = sorted(results, key=lambda x: x['iou'])
    results_sorted_best = sorted(results, key=lambda x: x['iou'], reverse=True)
    
    # Print worst N images
    print(f"\n{'='*80}")
    print(f"WORST {save_worst_n} IMAGES (by IoU):")
    print(f"{'='*80}")
    for i, result in enumerate(results_sorted_worst[:save_worst_n]):
        print(f"{i+1}. {result['image_filename']}: IoU={result['iou']:.4f}, F1={result['dice']:.4f}")
    
    # Print best N images
    print(f"\n{'='*80}")
    print(f"BEST {save_best_n} IMAGES (by IoU):")
    print(f"{'='*80}")
    for i, result in enumerate(results_sorted_best[:save_best_n]):
        print(f"{i+1}. {result['image_filename']}: IoU={result['iou']:.4f}, F1={result['dice']:.4f}")
    
    # Save worst and best images with visualizations
    os.makedirs(output_dir, exist_ok=True)
    
    # Save worst images
    worst_dir = os.path.join(output_dir, 'worst')
    os.makedirs(worst_dir, exist_ok=True)
    
    print(f"\nSaving worst {save_worst_n} images to {worst_dir}/...")
    for i, result in enumerate(results_sorted_worst[:save_worst_n]):
        save_visualization(result, worst_dir, f"worst_{i+1:03d}")
    
    # Save best images
    best_dir = os.path.join(output_dir, 'best')
    os.makedirs(best_dir, exist_ok=True)
    
    print(f"Saving best {save_best_n} images to {best_dir}/...")
    for i, result in enumerate(results_sorted_best[:save_best_n]):
        save_visualization(result, best_dir, f"best_{i+1:03d}")
    
    # Save summary CSV
    import pandas as pd
    summary_data = []
    for i, result in enumerate(results_sorted_worst):
        summary_data.append({
            'rank': i+1,
            'image_filename': result['image_filename'],
            'iou': result['iou'],
            'f1_dice': result['dice']
        })
    
    df = pd.DataFrame(summary_data)
    csv_path = os.path.join(output_dir, 'accuracy_results.csv')
    df.to_csv(csv_path, index=False)
    print(f"\nSaved summary to {csv_path}")
    
    print(f"\n{'='*80}")
    print(f"Results saved in: {output_dir}/")
    print(f"Worst image: {results_sorted_worst[0]['image_filename']} (IoU: {results_sorted_worst[0]['iou']:.4f})")
    print(f"Best image: {results_sorted_best[0]['image_filename']} (IoU: {results_sorted_best[0]['iou']:.4f})")
    print(f"{'='*80}")


def save_visualization(result, output_dir, prefix):
    """Save visualization of prediction vs ground truth"""
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
    filename = result['image_filename']
    output_path = os.path.join(output_dir, f"{prefix}_{filename}")
    cv2.imwrite(output_path, combined)
    
    # Also save individual masks
    cv2.imwrite(output_path.replace('.jpg', '_gt_mask.jpg').replace('.png', '_gt_mask.png'), gt_mask * 255)
    cv2.imwrite(output_path.replace('.jpg', '_pred_mask.jpg').replace('.png', '_pred_mask.png'), pred_mask_binary * 255)


def main():
    parser = argparse.ArgumentParser(description='Evaluate CULane dataset and find worst/best accuracy images')
    parser.add_argument("--dataset_dir", type=str, required=True,
                       help="Directory containing images and .lines.txt files")
    parser.add_argument("--model", type=str, default='./log/best_model.pth',
                       help="Model path")
    parser.add_argument("--model_type", type=str, default='ENet', help="Model type")
    parser.add_argument("--width", type=int, default=512, help="Resize width")
    parser.add_argument("--height", type=int, default=256, help="Resize height")
    parser.add_argument("--worst_n", type=int, default=10,
                       help="Number of worst accuracy images to save")
    parser.add_argument("--best_n", type=int, default=10,
                       help="Number of best accuracy images to save")
    parser.add_argument("--output", type=str, default='./culane_eval_results',
                       help="Output directory for results")
    
    args = parser.parse_args()
    
    evaluate_culane_dataset(
        dataset_dir=args.dataset_dir,
        model_path=args.model,
        model_type=args.model_type,
        resize_width=args.width,
        resize_height=args.height,
        save_worst_n=args.worst_n,
        save_best_n=args.best_n,
        output_dir=args.output
    )


if __name__ == "__main__":
    main()

