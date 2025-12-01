# coding: utf-8
"""
Kalman Filter Fusion Module
Combines ENet detections with Kalman filter predictions for improved lane tracking
"""
import numpy as np
import cv2
from model.utils.kalman_filter import MultiLaneKalmanTracker


class KalmanLaneFusion:
    """
    Fuses ENet binary/instance segmentation with Kalman filter predictions
    to produce smoother, more robust lane detections
    """
    
    def __init__(self, 
                 max_lanes=4,
                 n_points=10,
                 process_noise=0.03,
                 measurement_noise=0.3,
                 fusion_weight=0.7):
        """
        Initialize Kalman fusion system
        
        Args:
            max_lanes: Maximum number of lanes to track
            n_points: Number of points per lane
            process_noise: Process noise for Kalman (lower = more trust in model)
            measurement_noise: Measurement noise for Kalman (lower = more trust in ENet)
            fusion_weight: Weight for Kalman prediction vs ENet (0.5 = equal, >0.5 = more Kalman)
        """
        self.tracker = MultiLaneKalmanTracker(
            max_lanes=max_lanes,
            n_points=n_points,
            process_noise=process_noise,
            measurement_noise=measurement_noise
        )
        self.fusion_weight = fusion_weight
        self.frame_count = 0
    
    def fuse_detections(self, binary_mask, instance_mask=None, enet_confidence=1.0):
        """
        Fuse ENet detections with Kalman filter predictions
        
        Args:
            binary_mask: Binary segmentation mask from ENet (H, W) in 0-255 format
            instance_mask: Instance segmentation mask from ENet (H, W, 3) or None
            enet_confidence: Confidence in ENet detection (0-1)
        Returns:
            Dictionary with:
                - 'binary_mask': Fused binary mask
                - 'instance_mask': Fused instance mask (if available)
                - 'lanes': Tracked lane information
                - 'predicted_mask': Mask from Kalman predictions only
        """
        self.frame_count += 1
        
        # Convert to numpy if needed
        if isinstance(binary_mask, np.ndarray):
            binary_np = binary_mask.astype(np.uint8)
        else:
            binary_np = (binary_mask.numpy() * 255).astype(np.uint8) if hasattr(binary_mask, 'numpy') else binary_mask
        
        # Ensure binary mask is in correct format (0-255)
        if binary_np.max() <= 1:
            binary_np = (binary_np * 255).astype(np.uint8)
        
        h, w = binary_np.shape[:2]
        
        # Calculate measurement weight based on confidence and frame history
        # CRITICAL: In first frame, use ONLY ENet (no Kalman prediction possible)
        # Early frames: trust Kalman less (more ENet)
        # Stable tracking: trust Kalman more (smoother)
        if self.frame_count == 1:
            # First frame: only initialize, don't use predictions
            measurement_weight = 1.0  # Full trust in ENet, no Kalman
        elif self.frame_count < 5:
            measurement_weight = 0.9  # Still mostly ENet
        else:
            measurement_weight = enet_confidence
        
        # Update Kalman tracker
        try:
            tracked_lanes = self.tracker.update(
                binary_np,
                instance_mask,
                measurement_weight=measurement_weight
            )
        except Exception as e:
            print(f"Warning: Kalman tracker error: {e}. Using ENet detections only.")
            tracked_lanes = {}
        
        # Create predicted mask from Kalman
        predicted_mask = np.zeros((h, w), dtype=np.uint8)
        
        for lane_idx, lane_info in tracked_lanes.items():
            points = lane_info.get('points', None)
            if points is not None and len(points) >= 2:
                try:
                    # Validate points before drawing
                    points_array = np.array(points)
                    if len(points_array.shape) == 2 and points_array.shape[1] == 2:
                        # Check if points are ordered correctly (monotonic in y)
                        y_coords = points_array[:, 1]
                        if len(y_coords) > 1 and np.all(np.diff(y_coords) >= 0):
                            # Draw predicted lane
                            points_int = points_array.astype(np.int32)
                            for i in range(len(points_int) - 1):
                                pt1 = tuple(points_int[i])
                                pt2 = tuple(points_int[i + 1])
                                # Validate points are within image bounds
                                if (0 <= pt1[0] < w and 0 <= pt1[1] < h and
                                    0 <= pt2[0] < w and 0 <= pt2[1] < h):
                                    # Check for reasonable line length (avoid drawing very long lines)
                                    line_length = np.sqrt((pt2[0] - pt1[0])**2 + (pt2[1] - pt1[1])**2)
                                    if line_length < h * 1.5:  # Reasonable maximum
                                        cv2.line(predicted_mask, pt1, pt2, 255, 3)
                except Exception as e:
                    print(f"Warning: Error drawing lane {lane_idx}: {e}")
                    continue
        
        # Fuse masks: combine ENet detection with Kalman prediction
        # CRITICAL: On first frame, use ONLY ENet (no Kalman predictions yet)
        if self.frame_count == 1 or len(tracked_lanes) == 0:
            # First frame or no tracked lanes: use only ENet
            kalman_weight = 0.0
            enet_weight = 1.0
        else:
            # If measurement_weight is high (trust ENet), use more of ENet
            # If measurement_weight is low (trust Kalman), use more of Kalman
            # Reduce Kalman weight significantly in early frames
            kalman_weight_factor = 0.3 if self.frame_count < 5 else 0.5
            kalman_weight = (1.0 - measurement_weight) * kalman_weight_factor
            enet_weight = 1.0 - kalman_weight
        
        # Normalize weights
        total_weight = kalman_weight + enet_weight
        if total_weight > 0:
            kalman_weight /= total_weight
            enet_weight /= total_weight
        else:
            kalman_weight = 0.0
            enet_weight = 1.0
        
        # Binary mask fusion
        enet_binary = (binary_np > 127).astype(np.float32)
        kalman_binary = (predicted_mask > 127).astype(np.float32)
        
        # If no valid Kalman predictions or first frame, use only ENet
        if kalman_weight == 0.0 or np.sum(kalman_binary) == 0:
            fused_binary = enet_binary.astype(np.uint8) * 255
        else:
            fused_binary = (enet_weight * enet_binary + kalman_weight * kalman_binary)
            fused_binary = (fused_binary > 0.5).astype(np.uint8) * 255
        
        # Instance mask fusion (if available)
        fused_instance = None
        if instance_mask is not None:
            if len(instance_mask.shape) == 3:
                fused_instance = instance_mask.copy().astype(np.float32)
                # Enhance instance mask with Kalman predictions
                for lane_idx, lane_info in tracked_lanes.items():
                    if lane_info.get('matched', False):
                        points = lane_info['points']
                        if len(points) >= 2:
                            # Create instance channel enhancement
                            instance_channel = np.zeros((h, w), dtype=np.float32)
                            points_int = points.astype(np.int32)
                            for i in range(len(points_int) - 1):
                                cv2.line(instance_channel,
                                        tuple(points_int[i]),
                                        tuple(points_int[i + 1]),
                                        1.0, 5)
                            
                            # Blend with instance mask (simplified - would need proper channel assignment)
                            if instance_mask.shape[2] > lane_idx:
                                fused_instance[:, :, lane_idx] = (
                                    enet_weight * instance_mask[:, :, lane_idx] +
                                    kalman_weight * instance_channel
                                )
        
        return {
            'binary_mask': fused_binary,
            'instance_mask': fused_instance if fused_instance is not None else instance_mask,
            'lanes': tracked_lanes,
            'predicted_mask': predicted_mask,
            'enet_mask': binary_np,
            'fusion_weights': {
                'kalman': kalman_weight,
                'enet': enet_weight
            }
        }
    
    def reset(self):
        """Reset the tracker (useful for new video sequences)"""
        self.tracker = MultiLaneKalmanTracker(
            max_lanes=self.tracker.max_lanes,
            n_points=self.tracker.n_points,
            process_noise=self.tracker.filters[0].kf.processNoiseCov[0, 0],
            measurement_noise=self.tracker.filters[0].kf.measurementNoiseCov[0, 0]
        )
        self.frame_count = 0
    
    def get_tracked_lanes_only(self):
        """Get only the tracked lanes without fusion (pure Kalman prediction)"""
        return self.tracker.get_all_tracked_lanes()

