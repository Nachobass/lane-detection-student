# coding: utf-8
"""
Kalman Filter for Lane Tracking
Combines temporal prediction with ENet detection for smoother lane tracking
"""
import numpy as np
import cv2


class LaneKalmanFilter:
    """
    Kalman Filter for tracking individual lane lines across video frames.
    Tracks lane position and velocity to predict and smooth lane detections.
    """
    
    def __init__(self, n_points=10, process_noise=0.03, measurement_noise=0.3):
        """
        Initialize Kalman Filter for lane tracking
        
        Args:
            n_points: Number of points to track along the lane (vertical sampling)
            process_noise: Process noise covariance (how much we trust the model)
            measurement_noise: Measurement noise covariance (how much we trust ENet)
        """
        self.n_points = n_points
        self.initialized = False
        
        # Kalman Filter state: [x1, vx1, x2, vx2, ..., xn, vxn] for n points
        # Each point has position (x) and velocity (vx)
        state_dim = n_points * 2  # position + velocity for each point
        
        # Create Kalman Filter
        self.kf = cv2.KalmanFilter(state_dim, n_points)
        
        # State transition matrix (A): predicts next state
        # For each point: x_new = x_old + vx * dt, vx_new = vx_old (constant velocity model)
        dt = 1.0  # time step (assume 1 frame = 1 time unit)
        self.kf.transitionMatrix = np.eye(state_dim, dtype=np.float32)
        for i in range(n_points):
            idx = i * 2
            self.kf.transitionMatrix[idx, idx + 1] = dt  # x = x + v*dt
        
        # Measurement matrix (H): maps state to measurements (we only measure position)
        self.kf.measurementMatrix = np.zeros((n_points, state_dim), dtype=np.float32)
        for i in range(n_points):
            self.kf.measurementMatrix[i, i * 2] = 1.0  # measure position only
        
        # Process noise covariance (Q): uncertainty in process model
        self.kf.processNoiseCov = np.eye(state_dim, dtype=np.float32) * process_noise
        for i in range(n_points):
            self.kf.processNoiseCov[i * 2 + 1, i * 2 + 1] = process_noise * 0.5  # velocity noise
        
        # Measurement noise covariance (R): uncertainty in measurements
        self.kf.measurementNoiseCov = np.eye(n_points, dtype=np.float32) * measurement_noise
        
        # Error covariance (P): initial uncertainty
        self.kf.errorCovPost = np.eye(state_dim, dtype=np.float32) * 1.0
        
        # Control matrix (B): not used
        self.kf.controlMatrix = None
        
        self.last_measurement = None
        self.frame_count = 0
    
    def initialize(self, lane_points):
        """
        Initialize the filter with first measurement
        
        Args:
            lane_points: Array of (x, y) coordinates for lane points
        """
        if len(lane_points) < 2:
            return False
        
        # Extract x coordinates at evenly spaced y coordinates
        y_coords = np.linspace(lane_points[0][1], lane_points[-1][1], self.n_points)
        x_coords = np.interp(y_coords, 
                            [p[1] for p in lane_points], 
                            [p[0] for p in lane_points])
        
        # Initialize state: positions and zero velocities
        initial_state = np.zeros(self.kf.statePre.shape[0], dtype=np.float32)
        for i, x in enumerate(x_coords):
            initial_state[i * 2] = x  # position
            initial_state[i * 2 + 1] = 0.0  # initial velocity = 0
        
        self.kf.statePre = initial_state
        self.kf.statePost = initial_state.copy()
        
        self.last_measurement = np.array(x_coords, dtype=np.float32)
        self.y_coords = y_coords
        self.initialized = True
        self.frame_count = 1
        
        return True
    
    def predict(self):
        """
        Predict next state based on motion model
        Returns predicted lane points
        """
        if not self.initialized:
            return None
        
        # Predict next state
        prediction = self.kf.predict()
        
        # Extract predicted positions
        predicted_x = prediction[::2]  # every other element (positions)
        predicted_points = np.array([[x, y] for x, y in zip(predicted_x, self.y_coords)])
        
        return predicted_points
    
    def update(self, lane_points, measurement_weight=1.0):
        """
        Update filter with new measurement from ENet
        
        Args:
            lane_points: Array of (x, y) coordinates for detected lane points
            measurement_weight: Weight to give to this measurement (0-1)
        Returns:
            Corrected/smoothed lane points
        """
        if len(lane_points) < 2:
            # If measurement is invalid, just use prediction
            return self.predict()
        
        # Extract x coordinates at same y positions
        x_coords = np.interp(self.y_coords,
                            [p[1] for p in lane_points],
                            [p[0] for p in lane_points])
        
        measurement = np.array(x_coords, dtype=np.float32).reshape(-1, 1)
        
        # Adjust measurement noise based on weight
        # Lower weight = less trust in measurement = higher noise
        original_noise = self.kf.measurementNoiseCov.copy()
        if measurement_weight < 1.0:
            self.kf.measurementNoiseCov = original_noise / (measurement_weight + 0.1)
        
        # Update filter
        self.kf.correct(measurement)
        
        # Restore original noise
        self.kf.measurementNoiseCov = original_noise
        
        # Get corrected state
        corrected_state = self.kf.statePost
        corrected_x = corrected_state[::2]
        corrected_points = np.array([[x, y] for x, y in zip(corrected_x, self.y_coords)])
        
        self.last_measurement = measurement.flatten()
        self.frame_count += 1
        
        return corrected_points
    
    def get_velocity(self):
        """Get current velocity estimates for all points"""
        if not self.initialized:
            return None
        state = self.kf.statePost
        velocities = state[1::2]  # every other element starting from 1 (velocities)
        return velocities


class MultiLaneKalmanTracker:
    """
    Track multiple lanes using separate Kalman Filters for each lane
    """
    
    def __init__(self, max_lanes=4, n_points=10, process_noise=0.03, measurement_noise=0.3):
        """
        Initialize multi-lane tracker
        
        Args:
            max_lanes: Maximum number of lanes to track
            n_points: Number of points per lane
            process_noise: Process noise for Kalman filter
            measurement_noise: Measurement noise for Kalman filter
        """
        self.max_lanes = max_lanes
        self.n_points = n_points
        self.filters = [LaneKalmanFilter(n_points, process_noise, measurement_noise) 
                       for _ in range(max_lanes)]
        self.active_lanes = []  # List of active lane indices
        self.min_iou_for_match = 0.3  # Minimum IoU to match lanes
    
    def extract_lanes_from_mask(self, binary_mask, instance_mask=None):
        """
        Extract lane lines from binary segmentation mask
        
        Args:
            binary_mask: Binary segmentation mask from ENet
            instance_mask: Optional instance segmentation mask
        Returns:
            List of lane point arrays
        """
        lanes = []
        
        if instance_mask is not None and len(instance_mask.shape) == 3:
            # Use instance segmentation to separate lanes
            h, w = binary_mask.shape[:2]
            y_samples = np.linspace(h * 0.6, h - 1, self.n_points, dtype=int)
            
            for instance_id in range(instance_mask.shape[2]):
                instance = instance_mask[:, :, instance_id]
                if instance.max() < 0.5:  # Skip weak detections
                    continue
                
                # Extract points for this instance
                lane_points = []
                for y in y_samples:
                    row = instance[y, :]
                    lane_pixels = np.where(row > 0.5)[0]
                    if len(lane_pixels) > 0:
                        x = int(np.mean(lane_pixels))
                        lane_points.append([x, y])
                
                if len(lane_points) >= 3:
                    lanes.append(np.array(lane_points))
        else:
            # Use binary mask and extract lanes using a simpler, more robust method
            h, w = binary_mask.shape
            y_samples = np.linspace(int(h * 0.6), h - 1, self.n_points, dtype=int)
            
            # Clean up mask with morphological operations
            kernel = np.ones((5, 5), np.uint8)
            cleaned_mask = cv2.morphologyEx(binary_mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
            
            # Extract lanes by scanning horizontal rows and grouping continuous segments
            lane_segments = {}  # Track lane segments across rows
            
            for y_idx, y in enumerate(y_samples):
                row = cleaned_mask[y, :]
                
                # Find all segments (continuous white regions) in this row
                segments = []
                in_segment = False
                start_x = 0
                
                for x in range(w):
                    if row[x] > 0 and not in_segment:
                        # Start of a new segment
                        start_x = x
                        in_segment = True
                    elif row[x] == 0 and in_segment:
                        # End of segment
                        end_x = x - 1
                        center_x = (start_x + end_x) // 2
                        width = end_x - start_x
                        if width > 3:  # Minimum width threshold
                            segments.append(center_x)
                        in_segment = False
                
                # Handle segment that extends to edge
                if in_segment:
                    center_x = (start_x + w - 1) // 2
                    segments.append(center_x)
                
                # Match segments to existing lanes or create new ones
                for seg_x in segments:
                    matched = False
                    for lane_id, lane_data in lane_segments.items():
                        if len(lane_data['points']) > 0:
                            last_x = lane_data['points'][-1][0]
                            # Match if x position is similar (within threshold)
                            if abs(seg_x - last_x) < w * 0.15:
                                lane_data['points'].append([seg_x, y])
                                matched = True
                                break
                    
                    if not matched:
                        # Create new lane
                        lane_id = len(lane_segments)
                        lane_segments[lane_id] = {'points': [[seg_x, y]]}
            
            # Convert to final lane format and interpolate missing points
            for lane_id, lane_data in lane_segments.items():
                lane_points = lane_data['points']
                if len(lane_points) >= 3:
                    # Sort by y coordinate (bottom to top)
                    lane_points = sorted(lane_points, key=lambda p: p[1])
                    
                    # Validate lane monotonicity - check for backwards movement
                    x_coords = [p[0] for p in lane_points]
                    y_coords = [p[1] for p in lane_points]
                    
                    # Check if lane has reasonable continuity (no large jumps)
                    valid_lane = True
                    for i in range(1, len(x_coords)):
                        dx = abs(x_coords[i] - x_coords[i-1])
                        dy = abs(y_coords[i] - y_coords[i-1])
                        if dy > 0 and dx / dy > 2.0:  # Too much horizontal movement
                            valid_lane = False
                            break
                    
                    if not valid_lane:
                        continue
                    
                    # Interpolate to get points at all y_samples
                    # Only interpolate within valid range (no extrapolation)
                    interpolated_lane = []
                    for y in y_samples:
                        if y_coords[0] <= y <= y_coords[-1]:
                            # Linear interpolation
                            x = np.interp(y, y_coords, x_coords)
                            interpolated_lane.append([int(x), y])
                        elif y < y_coords[0]:
                            # Below first point - use first point
                            interpolated_lane.append([x_coords[0], y])
                        else:
                            # Above last point - use last point (but don't extrapolate far)
                            if y - y_coords[-1] < h * 0.1:  # Only small extrapolation
                                interpolated_lane.append([x_coords[-1], y])
                    
                    # Final validation: ensure lane has at least 5 points and is monotonic
                    if len(interpolated_lane) >= 5:
                        # Check final monotonicity
                        final_x = [p[0] for p in interpolated_lane]
                        final_y = [p[1] for p in interpolated_lane]
                        
                        # Lane should be relatively smooth
                        x_variance = np.var(final_x)
                        if x_variance < w * w:  # Reasonable variance
                            lanes.append(np.array(interpolated_lane))
        
        return lanes
    
    def iou_lanes(self, lane1, lane2):
        """Calculate IoU between two lanes (simplified: based on area overlap)"""
        if len(lane1) < 2 or len(lane2) < 2:
            return 0.0
        
        try:
            # Get bounding box
            all_y = np.concatenate([lane1[:, 1], lane2[:, 1]])
            all_x = np.concatenate([lane1[:, 0], lane2[:, 0]])
            
            min_y = int(max(0, np.min(all_y)))
            max_y = int(np.max(all_y))
            min_x = int(max(0, np.min(all_x)))
            max_x = int(np.max(all_x))
            
            # Add padding
            width = max(512, max_x - min_x + 20)
            height = max_y - min_y + 1
            
            # Create masks for each lane
            mask1 = np.zeros((height, width), dtype=np.uint8)
            mask2 = np.zeros_like(mask1)
            
            for i in range(len(lane1) - 1):
                pt1 = (int(lane1[i, 0] - min_x + 10), int(lane1[i, 1] - min_y))
                pt2 = (int(lane1[i+1, 0] - min_x + 10), int(lane1[i+1, 1] - min_y))
                if 0 <= pt1[0] < width and 0 <= pt1[1] < height and \
                   0 <= pt2[0] < width and 0 <= pt2[1] < height:
                    cv2.line(mask1, pt1, pt2, 255, 3)
            
            for i in range(len(lane2) - 1):
                pt1 = (int(lane2[i, 0] - min_x + 10), int(lane2[i, 1] - min_y))
                pt2 = (int(lane2[i+1, 0] - min_x + 10), int(lane2[i+1, 1] - min_y))
                if 0 <= pt1[0] < width and 0 <= pt1[1] < height and \
                   0 <= pt2[0] < width and 0 <= pt2[1] < height:
                    cv2.line(mask2, pt1, pt2, 255, 3)
            
            intersection = cv2.bitwise_and(mask1, mask2)
            union = cv2.bitwise_or(mask1, mask2)
            
            intersection_area = np.sum(intersection > 0)
            union_area = np.sum(union > 0)
            
            if union_area == 0:
                return 0.0
            
            return intersection_area / union_area
        except Exception as e:
            # Fallback: use distance-based matching
            if len(lane1) == len(lane2):
                distances = np.sqrt(np.sum((lane1 - lane2)**2, axis=1))
                avg_distance = np.mean(distances)
                # Convert distance to similarity (closer = more similar)
                max_expected_distance = 50  # pixels
                similarity = max(0, 1 - (avg_distance / max_expected_distance))
                return similarity
            return 0.0
    
    def update(self, binary_mask, instance_mask=None, measurement_weight=1.0):
        """
        Update all lane trackers with new detections
        
        Args:
            binary_mask: Binary segmentation mask from ENet
            instance_mask: Optional instance segmentation mask
            measurement_weight: Weight for measurements (0-1)
        Returns:
            Dictionary with tracked lanes
        """
        # Extract lanes from current frame
        detected_lanes = self.extract_lanes_from_mask(binary_mask, instance_mask)
        
        # Match detected lanes to existing tracks
        matched = [False] * len(self.active_lanes)
        matched_detections = [False] * len(detected_lanes)
        
        results = {}
        
        # Try to match existing tracks
        for i, lane_idx in enumerate(self.active_lanes):
            if matched[i]:
                continue
            
            best_iou = 0
            best_detection_idx = -1
            
            # Predict current position
            predicted = self.filters[lane_idx].predict()
            if predicted is None:
                continue
            
            # Find best matching detection
            for j, detected_lane in enumerate(detected_lanes):
                if matched_detections[j]:
                    continue
                
                iou = self.iou_lanes(predicted, detected_lane)
                if iou > best_iou and iou > self.min_iou_for_match:
                    best_iou = iou
                    best_detection_idx = j
            
            if best_detection_idx >= 0:
                # Update existing track
                corrected = self.filters[lane_idx].update(
                    detected_lanes[best_detection_idx],
                    measurement_weight
                )
                results[lane_idx] = {
                    'points': corrected,
                    'predicted': predicted,
                    'matched': True,
                    'confidence': best_iou
                }
                matched[i] = True
                matched_detections[best_detection_idx] = True
            else:
                # No match - use prediction only (lower weight) but validate it first
                if predicted is not None:
                    # Validate predicted lane
                    pred_y = predicted[:, 1]
                    pred_x = predicted[:, 0]
                    
                    # Check if prediction is reasonable
                    is_valid = True
                    if len(pred_y) > 1:
                        # Should be monotonic in y
                        if not np.all(np.diff(pred_y) >= 0):
                            is_valid = False
                        # X should not have extreme jumps
                        x_diffs = np.abs(np.diff(pred_x))
                        if np.any(x_diffs > binary_mask.shape[1] * 0.3):
                            is_valid = False
                    
                    if is_valid:
                        results[lane_idx] = {
                            'points': predicted,
                            'predicted': predicted,
                            'matched': False,
                            'confidence': 0.3
                        }
                    # If prediction is invalid, don't add it to results (lane will be lost)
        
        # Initialize new tracks for unmatched detections
        for j, detected_lane in enumerate(detected_lanes):
            if matched_detections[j]:
                continue
            
            # Validate lane before initializing
            if len(detected_lane) < 3:
                continue
            
            # Check lane validity: should be monotonic in y, reasonable x spread
            lane_y = detected_lane[:, 1]
            lane_x = detected_lane[:, 0]
            
            # Y should be increasing (top to bottom in image coordinates)
            if not np.all(np.diff(lane_y) >= 0):
                continue  # Skip non-monotonic lanes
            
            # X should have reasonable spread (not too erratic)
            x_std = np.std(lane_x)
            if x_std > binary_mask.shape[1] * 0.5:  # Too much spread
                continue
            
            # Find free slot
            if len(self.active_lanes) < self.max_lanes:
                new_idx = len(self.active_lanes)
                self.active_lanes.append(new_idx)
                if self.filters[new_idx].initialize(detected_lane):
                    results[new_idx] = {
                        'points': detected_lane,
                        'predicted': detected_lane,
                        'matched': True,
                        'confidence': 0.5
                    }
        
        # Remove inactive tracks (not seen for too long)
        # This would require tracking frame counts - simplified for now
        
        return results
    
    def get_all_tracked_lanes(self):
        """Get all currently tracked lanes"""
        results = {}
        for lane_idx in self.active_lanes:
            if self.filters[lane_idx].initialized:
                predicted = self.filters[lane_idx].predict()
                if predicted is not None:
                    results[lane_idx] = {
                        'points': predicted,
                        'confidence': 0.5
                    }
        return results

