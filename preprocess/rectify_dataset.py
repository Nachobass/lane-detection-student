# coding: utf-8
"""
Batch Dataset Rectification Script

Processes entire datasets to apply homography rectification
to all images and corresponding masks.
"""
import os
import argparse
import cv2
import numpy as np
from typing import Optional
from tqdm import tqdm
from pathlib import Path
from preprocess.homography_rectification import HomographyRectifier


def process_image_pair(image_path: str, 
                      output_image_path: str,
                      mask_path: Optional[str] = None,
                      output_mask_path: Optional[str] = None,
                      rectifier: HomographyRectifier = None) -> bool:
    """
    Process a single image (and optionally mask) with rectification
    
    Args:
        image_path: Path to input image
        output_image_path: Path to save rectified image
        mask_path: Path to input mask (optional)
        output_mask_path: Path to save rectified mask (optional)
        rectifier: HomographyRectifier instance
    
    Returns:
        True if successful, False otherwise
    """
    try:
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            print(f"Warning: Could not load image {image_path}")
            return False
        
        # Rectify image
        rectified_image, homography = rectifier.rectify_image(image)
        
        # Save rectified image
        os.makedirs(os.path.dirname(output_image_path), exist_ok=True)
        cv2.imwrite(output_image_path, rectified_image)
        
        # Process mask if provided
        if mask_path is not None and output_mask_path is not None:
            if os.path.exists(mask_path):
                mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                if mask is not None:
                    rectified_mask = rectifier.rectify_mask(mask, homography)
                    os.makedirs(os.path.dirname(output_mask_path), exist_ok=True)
                    cv2.imwrite(output_mask_path, rectified_mask)
            else:
                # Create empty mask if not found
                h, w = rectified_image.shape[:2]
                empty_mask = np.zeros((h, w), dtype=np.uint8)
                os.makedirs(os.path.dirname(output_mask_path), exist_ok=True)
                cv2.imwrite(output_mask_path, empty_mask)
        
        return True
        
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return False


def process_tusimple_format(input_dir: str, 
                           output_dir: str,
                           rectifier: HomographyRectifier = None):
    """
    Process TuSimple format dataset
    
    Expected structure:
    input_dir/
        image/
            *.png
        gt_image_binary/
            *.png
        gt_image_instance/
            *.png
        train.txt / val.txt
    
    Args:
        input_dir: Input dataset directory
        output_dir: Output directory for rectified data
        rectifier: HomographyRectifier instance
    """
    if rectifier is None:
        rectifier = HomographyRectifier()
    
    # Create output structure
    output_image_dir = os.path.join(output_dir, 'image')
    output_binary_dir = os.path.join(output_dir, 'gt_image_binary')
    output_instance_dir = os.path.join(output_dir, 'gt_image_instance')
    
    os.makedirs(output_image_dir, exist_ok=True)
    os.makedirs(output_binary_dir, exist_ok=True)
    os.makedirs(output_instance_dir, exist_ok=True)
    
    # Process train.txt and val.txt
    for split_file in ['train.txt', 'val.txt']:
        input_split_file = os.path.join(input_dir, split_file)
        if not os.path.exists(input_split_file):
            continue
        
        output_split_file = os.path.join(output_dir, split_file)
        output_lines = []
        
        with open(input_split_file, 'r') as f:
            lines = f.readlines()
        
        print(f"Processing {split_file} with {len(lines)} entries...")
        
        for line in tqdm(lines, desc=f"Processing {split_file}"):
            parts = line.strip().split()
            if len(parts) < 3:
                continue
            
            # Original paths
            orig_image_path = parts[0]
            orig_binary_path = parts[1]
            orig_instance_path = parts[2]
            
            # Output paths
            rel_image_path = os.path.relpath(orig_image_path, input_dir)
            output_image_path = os.path.join(output_dir, rel_image_path)
            output_binary_path = os.path.join(output_dir, 
                                            os.path.relpath(orig_binary_path, input_dir))
            output_instance_path = os.path.join(output_dir,
                                              os.path.relpath(orig_instance_path, input_dir))
            
            # Process image
            if os.path.exists(orig_image_path):
                # Rectify image
                image = cv2.imread(orig_image_path)
                if image is not None:
                    rectified_image, homography = rectifier.rectify_image(image)
                    os.makedirs(os.path.dirname(output_image_path), exist_ok=True)
                    cv2.imwrite(output_image_path, rectified_image)
                    
                    # Rectify binary mask
                    if os.path.exists(orig_binary_path):
                        binary_mask = cv2.imread(orig_binary_path, cv2.IMREAD_GRAYSCALE)
                        if binary_mask is not None:
                            rectified_binary = rectifier.rectify_mask(binary_mask, homography)
                            os.makedirs(os.path.dirname(output_binary_path), exist_ok=True)
                            cv2.imwrite(output_binary_path, rectified_binary)
                    
                    # Rectify instance mask
                    if os.path.exists(orig_instance_path):
                        instance_mask = cv2.imread(orig_instance_path, cv2.IMREAD_UNCHANGED)
                        if instance_mask is not None:
                            rectified_instance = rectifier.rectify_mask(instance_mask, homography)
                            os.makedirs(os.path.dirname(output_instance_path), exist_ok=True)
                            cv2.imwrite(output_instance_path, rectified_instance)
            
            # Update paths in output file
            output_lines.append(f"{output_image_path} {output_binary_path} {output_instance_path}\n")
        
        # Write output split file
        with open(output_split_file, 'w') as f:
            f.writelines(output_lines)
        
        print(f"Saved {output_split_file}")


def main():
    parser = argparse.ArgumentParser(
        description='Rectify dataset images using homography transformation'
    )
    parser.add_argument('--input', type=str, required=True,
                       help='Input dataset directory')
    parser.add_argument('--output', type=str, required=True,
                       help='Output directory for rectified dataset')
    parser.add_argument('--canny_low', type=int, default=50,
                       help='Canny edge detection lower threshold')
    parser.add_argument('--canny_high', type=int, default=150,
                       help='Canny edge detection upper threshold')
    parser.add_argument('--hough_threshold', type=int, default=100,
                       help='Hough line detection threshold')
    
    args = parser.parse_args()
    
    # Create rectifier
    rectifier = HomographyRectifier(
        canny_low=args.canny_low,
        canny_high=args.canny_high,
        hough_threshold=args.hough_threshold
    )
    
    # Process dataset
    print(f"Processing dataset from {args.input} to {args.output}")
    process_tusimple_format(args.input, args.output, rectifier)
    print("Dataset rectification complete!")


if __name__ == '__main__':
    main()

