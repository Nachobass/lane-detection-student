"""
Generate CULane dataset for evaluation
Processes CULane format (images + .lines.txt files) and generates binary/instance masks
"""

import argparse
import os
import os.path as ops
import glob
import cv2
import numpy as np


def init_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--src_dir', type=str, help='Path to CULane dataset root (e.g., ./data/culane)')
    parser.add_argument('--list_file', type=str, help='Path to list file (e.g., ./data/culane/list/test.txt)')
    parser.add_argument('--output_dir', type=str, help='Output directory for processed data', default=None)
    return parser.parse_args()


def parse_lines_file(lines_file_path):
    """
    Parse CULane .lines.txt file
    Format: Each line is a lane with pairs of (x, y) coordinates
    Returns: List of lanes, each lane is a list of (x, y) tuples
    """
    lanes = []
    if not os.path.exists(lines_file_path):
        return lanes
    
    with open(lines_file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            # Parse coordinates: x1 y1 x2 y2 x3 y3 ...
            coords = line.split()
            if len(coords) < 2:
                continue
            
            # Convert to pairs of (x, y)
            lane_points = []
            for i in range(0, len(coords), 2):
                if i + 1 < len(coords):
                    try:
                        x = float(coords[i])
                        y = float(coords[i + 1])
                        # Skip invalid coordinates (negative x often means lane not visible)
                        if x >= 0 and y >= 0:
                            lane_points.append((int(x), int(y)))
                    except ValueError:
                        continue
            
            if len(lane_points) >= 2:  # Need at least 2 points for a lane
                lanes.append(lane_points)
    
    return lanes


def create_masks_from_lanes(image_shape, lanes):
    """
    Create binary and instance masks from lane points
    """
    h, w = image_shape[:2]
    binary_mask = np.zeros((h, w), dtype=np.uint8)
    instance_mask = np.zeros((h, w), dtype=np.uint8)
    
    for lane_idx, lane_points in enumerate(lanes):
        if len(lane_points) < 2:
            continue
        
        # Convert to numpy array for cv2.polylines
        lane_pts = np.array([lane_points], dtype=np.int32)
        
        # Draw on binary mask (all lanes in white)
        cv2.polylines(binary_mask, lane_pts, isClosed=False, color=255, thickness=5)
        
        # Draw on instance mask (each lane with different color)
        # Use lane_idx * 50 + 20 to get different shades
        instance_color = min(255, lane_idx * 50 + 20)
        cv2.polylines(instance_mask, lane_pts, isClosed=False, color=instance_color, thickness=5)
    
    return binary_mask, instance_mask


def process_culane_dataset(src_dir, list_file, output_dir=None):
    """
    Process CULane dataset and generate masks
    """
    if output_dir is None:
        output_dir = ops.join(src_dir, 'processed')
    
    # Create output directories
    gt_image_dir = ops.join(output_dir, 'gt_image')
    gt_binary_dir = ops.join(output_dir, 'gt_binary_image')
    gt_instance_dir = ops.join(output_dir, 'gt_instance_image')
    
    os.makedirs(gt_image_dir, exist_ok=True)
    os.makedirs(gt_binary_dir, exist_ok=True)
    os.makedirs(gt_instance_dir, exist_ok=True)
    
    # Read list file
    image_paths = []
    with open(list_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                # CULane list files have paths like: /driver_100_30frame/05251517_0433.MP4/00000.jpg
                # Remove leading slash if present
                if line.startswith('/'):
                    line = line[1:]
                image_paths.append(line)
    
    print(f"Found {len(image_paths)} images in list file")
    
    # Check which images have already been processed
    processed_indices = set()
    if os.path.exists(gt_image_dir):
        for f in os.listdir(gt_image_dir):
            if f.endswith('.png'):
                try:
                    idx = int(f.replace('.png', ''))
                    processed_indices.add(idx)
                except ValueError:
                    pass
    
    print(f"Found {len(processed_indices)} already processed images. Continuing from index {len(processed_indices)}...")
    
    # Process each image
    processed_count = len(processed_indices)
    test_txt_lines = []
    
    # Load existing test.txt if it exists
    test_txt_path = ops.join(output_dir, 'test.txt')
    if os.path.exists(test_txt_path):
        with open(test_txt_path, 'r') as f:
            test_txt_lines = [line.strip() for line in f if line.strip()]
        print(f"Loaded {len(test_txt_lines)} existing entries from test.txt")
    
    for img_idx, rel_image_path in enumerate(image_paths):
        if img_idx in processed_indices:
            image_name = f"{img_idx:04d}.png"
            gt_image_path = ops.join(gt_image_dir, image_name)
            gt_binary_path = ops.join(gt_binary_dir, image_name)
            gt_instance_path = ops.join(gt_instance_dir, image_name)
            
            # Only add if not already in test_txt_lines
            entry = f"{gt_image_path} {gt_binary_path} {gt_instance_path}"
            if entry not in test_txt_lines:
                test_txt_lines.append(entry)
            continue
        
        image_path = ops.join(src_dir, rel_image_path)
        
        lines_path = image_path.replace('.jpg', '.lines.txt')
        
        if not os.path.exists(image_path):
            print(f"Warning [{img_idx}]: Image not found: {image_path}")
            continue
        
        img = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if img is None:
            print(f"Warning [{img_idx}]: Could not load image: {image_path}")
            continue
        
        lanes = parse_lines_file(lines_path)
        
        binary_mask, instance_mask = create_masks_from_lanes(img.shape, lanes)
        
        image_name = f"{img_idx:04d}.png"
        gt_image_path = ops.join(gt_image_dir, image_name)
        gt_binary_path = ops.join(gt_binary_dir, image_name)
        gt_instance_path = ops.join(gt_instance_dir, image_name)
        
        cv2.imwrite(gt_image_path, img)
        cv2.imwrite(gt_binary_path, binary_mask)
        cv2.imwrite(gt_instance_path, instance_mask)
        
        test_txt_lines.append(f"{gt_image_path} {gt_binary_path} {gt_instance_path}")
        
        processed_count += 1
        if processed_count % 100 == 0:
            print(f"Processed {processed_count}/{len(image_paths)} images ({processed_count*100//len(image_paths)}%)...")
        elif processed_count % 10 == 0:
            print(f"Processed {processed_count}/{len(image_paths)} images...", end='\r')
    
    test_txt_path = ops.join(output_dir, 'test.txt')
    def get_index(line):
        # Extract index from path like ".../gt_image/0001.png"
        parts = line.split()
        if parts:
            filename = ops.basename(parts[0])
            try:
                return int(filename.replace('.png', ''))
            except:
                return 0
        return 0
    
    test_txt_lines_sorted = sorted(test_txt_lines, key=get_index)
    with open(test_txt_path, 'w') as f:
        for line in test_txt_lines_sorted:
            f.write(line + '\n')
    
    print(f"\nProcessing complete!")
    print(f"Processed {processed_count} images")
    print(f"Output directory: {output_dir}")
    print(f"Test file: {test_txt_path}")
    
    return output_dir


if __name__ == '__main__':
    args = init_args()
    
    if not args.src_dir or not args.list_file:
        print("Error: --src_dir and --list_file are required")
        print("Example:")
        print("  python culane_transform.py --src_dir ./data/culane --list_file ./data/culane/list/test.txt")
        exit(1)
    
    process_culane_dataset(args.src_dir, args.list_file, args.output_dir)

