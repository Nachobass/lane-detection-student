"""
Test script for processing image sequences with Kalman filter
Processes multiple images in sequence to test temporal tracking
"""
import time
import os
import sys
import glob
import argparse

import torch
from model.lanenet.LaneNet import LaneNet
from torchvision import transforms
import numpy as np
from PIL import Image
import cv2

# Import Kalman filter fusion module
try:
    from model.utils.kalman_fusion import KalmanLaneFusion
    KALMAN_AVAILABLE = True
except ImportError:
    KALMAN_AVAILABLE = False
    print("Warning: Kalman filter module not available.")

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def load_test_data(img_path, transform):
    img = Image.open(img_path)
    # Convert RGBA to RGB if necessary (PNG images may have alpha channel)
    if img.mode == 'RGBA':
        # Create white background and composite
        rgb_img = Image.new('RGB', img.size, (255, 255, 255))
        rgb_img.paste(img, mask=img.split()[3])  # Use alpha channel as mask
        img = rgb_img
    elif img.mode == 'L':
        # Convert grayscale to RGB
        img = img.convert('RGB')
    elif img.mode != 'RGB':
        # Convert any other mode to RGB
        img = img.convert('RGB')
    img = transform(img)
    return img


# Global variable to maintain consistent lane count across frames
_lane_count_history = []
_stable_lane_count = None


def process_single_frame(model, kalman_fusion, img_path, data_transform, 
                        resize_width, resize_height, frame_idx, output_dir, 
                        use_kalman=False, maintain_stable_count=True):
    """Process a single frame and return results"""
    
    # Load and preprocess image
    dummy_input = load_test_data(img_path, data_transform).to(DEVICE)
    dummy_input = torch.unsqueeze(dummy_input, dim=0)
    
    # Run model
    with torch.no_grad():
        outputs = model(dummy_input)
    
    # Get original image
    input_img = Image.open(img_path)
    # Convert RGBA to RGB if necessary
    if input_img.mode == 'RGBA':
        rgb_img = Image.new('RGB', input_img.size, (255, 255, 255))
        rgb_img.paste(input_img, mask=input_img.split()[3])
        input_img = rgb_img
    elif input_img.mode not in ('RGB', 'L'):
        input_img = input_img.convert('RGB')
    input_img = input_img.resize((resize_width, resize_height))
    input_array = np.array(input_img)
    
    # Process outputs
    instance_pred = torch.squeeze(outputs['instance_seg_logits'].detach().to('cpu')).numpy() * 255
    binary_pred = torch.squeeze(outputs['binary_seg_pred']).to('cpu').numpy()
    
    # Convert binary_pred to binary mask
    binary_mask = binary_pred.squeeze()
    if len(binary_mask.shape) > 2:
        binary_mask = binary_mask.squeeze()
    
    binary_mask = (binary_mask > 0).astype(np.uint8)
    
    # Resize if needed
    if binary_mask.shape != input_array.shape[:2]:
        binary_mask = cv2.resize(binary_mask, (input_array.shape[1], input_array.shape[0]), 
                                 interpolation=cv2.INTER_NEAREST)
    
    # Prepare instance mask
    instance_mask_resized = None
    if len(instance_pred.shape) == 3:
        instance_mask_resized = instance_pred.transpose((1, 2, 0))
        if instance_mask_resized.shape[:2] != input_array.shape[:2]:
            h, w = input_array.shape[:2]
            instance_mask_resized = cv2.resize(instance_mask_resized, (w, h), 
                                               interpolation=cv2.INTER_LINEAR)
        instance_mask_resized = instance_mask_resized / 255.0
    
    # Apply Kalman filter fusion if enabled 
    fusion_info = {}
    tracked_lanes_dict = {}  
    h, w = binary_mask.shape[:2]
    
    if use_kalman and kalman_fusion is not None:
        try:
            binary_mask_255 = (binary_mask * 255).astype(np.uint8)
            fusion_result = kalman_fusion.fuse_detections(
                binary_mask_255,
                instance_mask=instance_mask_resized,
                enet_confidence=0.7
            )
            
            binary_mask_fused = fusion_result['binary_mask']
            if binary_mask_fused is not None and binary_mask_fused.size > 0:
                binary_mask = (binary_mask_fused > 127).astype(np.uint8)
            
            tracked_lanes_dict = fusion_result.get('lanes', {})
            lane_lines_count = len(tracked_lanes_dict)
            fusion_info = {
                'weights': fusion_result['fusion_weights'],
                'lane_lines_count': lane_lines_count,
                'frame': kalman_fusion.frame_count
            }
        except Exception as e:
            print(f"Warning: Kalman fusion error in frame {frame_idx}: {e}")
            fusion_info = {}
    
    
    yellow_lines_count = 0
    detected_line_positions = []  
    
    global _lane_count_history, _stable_lane_count
    
    # BEST method: Use vertical projection on multiple regions to find ALL lane lines
    # This captures lines at different heights (closer and further)
    if np.sum(binary_mask) > 0:
        all_peaks_from_regions = []
        
        # Analyze multiple vertical regions at different heights
        region_starts = [int(h * 0.4), int(h * 0.5), int(h * 0.6), int(h * 0.7)]
        
        for region_start in region_starts:
            if region_start >= h:
                continue
            
            region = binary_mask[region_start:, :]
            x_projection = np.sum(region, axis=0)
            
            if np.max(x_projection) > 0:
                # Smooth the projection to reduce noise
                kernel_size = 5
                x_projection_smooth = np.convolve(x_projection, np.ones(kernel_size)/kernel_size, mode='same')
                
                # Find local maxima (peaks)
                max_val = np.max(x_projection_smooth)
                mean_val = np.mean(x_projection_smooth[x_projection_smooth > 0]) if np.any(x_projection_smooth > 0) else 0
                threshold = max(max_val * 0.08, mean_val * 0.25, 2)
                
                # Find peaks in this region
                peaks = []
                window = 4
                
                for x in range(window, len(x_projection_smooth) - window):
                    center_val = x_projection_smooth[x]
                    if center_val >= threshold:
                        is_max = True
                        for offset in range(-window, window + 1):
                            if offset != 0 and x_projection_smooth[x + offset] > center_val:
                                is_max = False
                                break
                        if is_max:
                            peaks.append(x)
                
                all_peaks_from_regions.extend(peaks)
        
        # Combine peaks from all regions and find consensus
        if len(all_peaks_from_regions) > 0:
            # Group peaks that are close (same line detected in multiple regions)
            all_peaks_from_regions = sorted(all_peaks_from_regions)
            grouped_peaks = []
            
            for peak in all_peaks_from_regions:
                if len(grouped_peaks) == 0:
                    grouped_peaks.append([peak])
                else:
                    added = False
                    for group in grouped_peaks:
                        if abs(peak - np.mean(group)) < 25:  # Same line across regions
                            group.append(peak)
                            added = True
                            break
                    if not added:
                        grouped_peaks.append([peak])
            
            # Get center of each group (each group = one lane line)
            detected_line_positions = [int(np.mean(group)) for group in grouped_peaks]
            
            # Final filter: remove positions too close together
            filtered_positions = []
            for pos in sorted(detected_line_positions):
                if len(filtered_positions) == 0 or pos - filtered_positions[-1] > 18:
                    filtered_positions.append(pos)
            
            detected_line_positions = filtered_positions
            yellow_lines_count = len(detected_line_positions)
        else:
            yellow_lines_count = 0
    else:
        yellow_lines_count = 0
    
    # Fallback 1: Use tracked lanes from Kalman if instance mask didn't work
    if yellow_lines_count == 0 and len(tracked_lanes_dict) > 0:
        # Get line positions from tracked lanes at bottom of image
        for lane_idx, lane_info in tracked_lanes_dict.items():
            points = lane_info.get('points', None)
            if points is not None and len(points) > 0:
                # Get x position at bottom of image (most reliable)
                points_array = np.array(points)
                if len(points_array.shape) == 2 and points_array.shape[1] == 2:
                    # Find point closest to bottom
                    y_coords = points_array[:, 1]
                    bottom_y = h - 1
                    closest_idx = np.argmin(np.abs(y_coords - bottom_y))
                    x_pos = int(points_array[closest_idx, 0])
                    if 0 <= x_pos < w:
                        detected_line_positions.append(x_pos)
        
        if len(detected_line_positions) > 0:
            detected_line_positions = sorted(set(detected_line_positions))
            yellow_lines_count = len(detected_line_positions)
    
    # Fallback 2: Extract lines from binary mask using multiple rows
    if yellow_lines_count == 0 and np.sum(binary_mask) > 0:
        # Analyze multiple rows for better detection
        y_samples = [int(h * 0.7), int(h * 0.8), int(h * 0.9), h - 1]
        all_segments = []
        
        for y in y_samples:
            if y >= h:
                continue
            row = binary_mask[y, :]
            
            # Find connected segments in this row
            segments = []
            in_segment = False
            segment_start = 0
            
            for x in range(len(row)):
                if row[x] > 0 and not in_segment:
                    segment_start = x
                    in_segment = True
                elif row[x] == 0 and in_segment:
                    segment_end = x - 1
                    segment_width = segment_end - segment_start + 1
                    # Lane lines are thin: 2-60 pixels wide (allow wider range)
                    if 2 <= segment_width <= 60:
                        segment_center = (segment_start + segment_end) // 2
                        segments.append(segment_center)
                    in_segment = False
            
            if in_segment:
                segment_width = len(row) - segment_start
                if 2 <= segment_width <= 60:
                    segment_center = (segment_start + len(row) - 1) // 2
                    segments.append(segment_center)
            
            all_segments.extend(segments)
        
        # Group segments that are close together (same line across rows)
        if len(all_segments) > 0:
            all_segments = sorted(all_segments)
            grouped = []
            
            for seg in all_segments:
                if len(grouped) == 0:
                    grouped.append([seg])
                else:
                    # Check if close to any existing group
                    added = False
                    for group in grouped:
                        if abs(seg - np.mean(group)) < 30:
                            group.append(seg)
                            added = True
                            break
                    if not added:
                        grouped.append([seg])
            
            # Get center of each group
            detected_line_positions = [int(np.mean(group)) for group in grouped]
            # Filter positions that are too close
            filtered_positions = []
            for pos in sorted(detected_line_positions):
                if len(filtered_positions) == 0 or pos - filtered_positions[-1] > 25:
                    filtered_positions.append(pos)
            
            detected_line_positions = filtered_positions
            yellow_lines_count = len(detected_line_positions)
    
    # Simple rule: Lanes = Yellow Lines - 1
    # if yellow_lines_count >= 4:
    #     lanes_count = 3  
    # elif yellow_lines_count == 3:
    #     lanes_count = 2  
    # elif yellow_lines_count == 2:
    #     lanes_count = 1  
    # elif yellow_lines_count == 1:
    #     lanes_count = 1  
    # else:
    #     lanes_count = 0
    lanes_count = max(1, yellow_lines_count - 1) if yellow_lines_count > 0 else 0
    
    # DISABLED: Temporal consensus was causing it to stabilize on wrong count
    # Just use direct calculation based on detected lines
    # lanes_count is already calculated above
    
    # Ensure reasonable bounds (1-4 lanes for highways)
    lanes_count = max(1, min(4, lanes_count)) if lanes_count > 0 else 0
    
    # Add lane count info to fusion_info (always, with or without Kalman)
    if 'weights' not in fusion_info:
        # Default weights when Kalman is not used
        fusion_info['weights'] = {'kalman': 0.0, 'enet': 1.0}
    if 'frame' not in fusion_info:
        # Default frame number when Kalman is not used
        fusion_info['frame'] = frame_idx
    fusion_info['lanes_count'] = lanes_count
    fusion_info['yellow_lines_detected'] = yellow_lines_count
    fusion_info['line_positions'] = detected_line_positions  # Store line positions for drawing
    
    # Convert to BGR for OpenCV
    if len(input_array.shape) == 3 and input_array.shape[2] == 3:
        overlay_image = cv2.cvtColor(input_array, cv2.COLOR_RGB2BGR)
    else:
        overlay_image = input_array.copy()
    
    # Create overlay with lane detections
    color_overlay = np.zeros_like(overlay_image)
    binary_mask_vis = (binary_mask * 255).astype(np.uint8) if binary_mask.max() <= 1 else binary_mask.astype(np.uint8)
    color_overlay[binary_mask_vis > 0] = [0, 255, 255]  # Yellow in BGR
    
    overlay_result = cv2.addWeighted(overlay_image, 0.65, color_overlay, 0.35, 0)
    
    # Draw individual detected lines as bright vertical lines
    # Draw each detected line from top to bottom where there are detections
    if len(detected_line_positions) > 0:
        for line_x in detected_line_positions:
            x_int = int(line_x)
            if 0 <= x_int < w:
                # Use a small window around the line position to find vertical extent
                window_size = 5
                x_min = max(0, x_int - window_size)
                x_max = min(w - 1, x_int + window_size)
                
                # Find vertical extent in this region
                region = binary_mask[:, x_min:x_max+1]
                line_pixels_y = np.where(np.any(region > 0, axis=1))[0]
                
                if len(line_pixels_y) > 0:
                    y_top = int(line_pixels_y[0])
                    y_bottom = int(line_pixels_y[-1])
                    
                    # Draw a bright vertical line at each detected position
                    cv2.line(overlay_result, 
                            (x_int, y_top),      
                            (x_int, y_bottom),   
                            (255, 0, 255),       # Magenta color in BGR (bright)
                            4)                   
                else:
                    cv2.line(overlay_result, 
                            (x_int, int(h * 0.6)),
                            (x_int, h - 1),
                            (255, 0, 255),
                            4)
    
    # Add frame info text
    info_text = f"Frame {frame_idx:04d}"
    if fusion_info:
        info_text += f" | Kalman: {fusion_info['weights']['kalman']:.2f}"
        yellow_lines = fusion_info.get('yellow_lines_detected', 0)
        # if yellow_lines > 0:
        #     info_text += f" | Lines: {yellow_lines} | Lanes: {fusion_info['lanes_count']}"
        # else:
        #     info_text += f" | Lanes: {fusion_info['lanes_count']}"
        info_text += f" | Lanes: {fusion_info['lanes_count']}"
    cv2.putText(overlay_result, info_text, (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    # Save outputs
    frame_dir = os.path.join(output_dir, f"frame_{frame_idx:04d}")
    os.makedirs(frame_dir, exist_ok=True)
    
    cv2.imwrite(os.path.join(frame_dir, 'input_overlay.jpg'), overlay_result)
    cv2.imwrite(os.path.join(frame_dir, 'binary_mask.jpg'), (binary_mask * 255).astype(np.uint8))
    
    if instance_mask_resized is not None:
        instance_output = (instance_mask_resized * 255).astype(np.uint8) if instance_mask_resized.max() <= 1.0 else instance_mask_resized
        cv2.imwrite(os.path.join(frame_dir, 'instance_mask.jpg'), instance_output)
    
    return overlay_result, fusion_info



def test_sequence():
    parser = argparse.ArgumentParser(description='Test lane detection on image sequence with Kalman filter')
    parser.add_argument("--img_dir", type=str, required=True, 
                       help="Directory containing images or pattern (e.g., './data/test_kalman/*.jpg')")
    parser.add_argument("--model_type", type=str, default='ENet', help="Model type")
    parser.add_argument("--model", type=str, default='./log/best_model.pth', help="Model path")
    parser.add_argument("--width", type=int, default=512, help="Resize width")
    parser.add_argument("--height", type=int, default=256, help="Resize height")
    parser.add_argument("--output", type=str, default='./test_output_sequence', help="Output directory")
    parser.add_argument("--freeze_backbone", action="store_true", help="Freeze encoder backbone")
    parser.add_argument("--use_kalman", action="store_true", help="Use Kalman filter")
    parser.add_argument("--kalman_process_noise", type=float, default=0.03, help="Kalman process noise")
    parser.add_argument("--kalman_measurement_noise", type=float, default=0.3, help="Kalman measurement noise")
    parser.add_argument("--max_lanes", type=int, default=4, help="Maximum lanes to track")
    
    args = parser.parse_args()
    
    os.makedirs(args.output, exist_ok=True)
    
    if os.path.isdir(args.img_dir):
        
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']
        image_paths = []
        for ext in image_extensions:
            image_paths.extend(glob.glob(os.path.join(args.img_dir, ext)))
        
        def natural_sort_key(path):
            import re
            filename = os.path.basename(path)
            numbers = re.findall(r'\d+', filename)
            return [int(n) for n in numbers] if numbers else [0]
        image_paths = sorted(image_paths, key=natural_sort_key)
    else:
        
        image_paths = glob.glob(args.img_dir)
        
        def natural_sort_key(path):
            import re
            filename = os.path.basename(path)
            numbers = re.findall(r'\d+', filename)
            return [int(n) for n in numbers] if numbers else [0]
        image_paths = sorted(image_paths, key=natural_sort_key)
    
    if len(image_paths) == 0:
        print(f"Error: No images found in {args.img_dir}")
        return
    
    print(f"Found {len(image_paths)} images to process")
    
    # Reset global lane count tracking for new sequence
    global _lane_count_history, _stable_lane_count
    _lane_count_history = []
    _stable_lane_count = None
    
    # Setup transforms
    data_transform = transforms.Compose([
        transforms.Resize((args.height, args.width)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    
    print("Loading model...")
    model = LaneNet(arch=args.model_type, freeze_encoder=args.freeze_backbone)
    state_dict = torch.load(args.model, map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.eval()
    model.to(DEVICE)
    print("Model loaded successfully")
    
    
    kalman_fusion = None
    if args.use_kalman and KALMAN_AVAILABLE:
        print("Initializing Kalman filter for lane tracking...")
        kalman_fusion = KalmanLaneFusion(
            max_lanes=args.max_lanes,
            process_noise=args.kalman_process_noise,
            measurement_noise=args.kalman_measurement_noise
        )
        print(f"Kalman filter initialized - Process noise: {args.kalman_process_noise}, "
              f"Measurement noise: {args.kalman_measurement_noise}")
    elif args.use_kalman and not KALMAN_AVAILABLE:
        print("Warning: Kalman filter requested but not available. Continuing without Kalman.")
    
    # Process each frame
    print("\nProcessing frames...")
    start_time = time.time()
    
    for frame_idx, img_path in enumerate(image_paths, 1):
        print(f"Processing frame {frame_idx}/{len(image_paths)}: {os.path.basename(img_path)}")
        
        try:
            overlay_result, fusion_info = process_single_frame(
                model, kalman_fusion, img_path, data_transform,
                args.width, args.height, frame_idx, args.output,
                use_kalman=args.use_kalman,
                maintain_stable_count=True
            )
            
            if fusion_info:
                yellow_lines = fusion_info.get('yellow_lines_detected', 0)
                print(f"  -> Frame {fusion_info['frame']}: "
                      f"Kalman weight: {fusion_info['weights']['kalman']:.3f}, "
                      f"ENet weight: {fusion_info['weights']['enet']:.3f}, "
                    #   f"Yellow lines: {yellow_lines}, Lanes: {fusion_info['lanes_count']}")
                      f"Lanes: {fusion_info['lanes_count']}")
        except Exception as e:
            print(f"Error processing frame {frame_idx}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    elapsed_time = time.time() - start_time
    print(f"\nCompleted! Processed {len(image_paths)} frames in {elapsed_time:.2f} seconds")
    print(f"Average time per frame: {elapsed_time/len(image_paths):.2f} seconds")
    print(f"\nResults saved in: {args.output}")
    print(f"Each frame has its own directory: frame_0001/, frame_0002/, etc.")


if __name__ == "__main__":
    test_sequence()

