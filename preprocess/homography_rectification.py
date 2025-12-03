# coding: utf-8
"""
Homography-based Image Rectification for Lane Detection

This module implements dynamic homography estimation to rectify images
by detecting edges, estimating horizon line with RANSAC, and computing
a stable homography that "straightens" the scene perspective.
"""
import cv2
import numpy as np
from typing import Tuple, Optional, List
import warnings


class HomographyRectifier:
    """
    Class for estimating and applying homography transformations
    to rectify road images for better lane detection
    """
    
    def __init__(self, 
                 canny_low=50, 
                 canny_high=150,
                 hough_threshold=100,
                 hough_min_line_length=100,
                 hough_max_line_gap=10,
                 ransac_threshold=1.0,
                 ransac_max_iters=2000):
        """
        Args:
            canny_low: Lower threshold for Canny edge detection
            canny_high: Upper threshold for Canny edge detection
            hough_threshold: Threshold for Hough line detection
            hough_min_line_length: Minimum line length for Hough
            hough_max_line_gap: Maximum gap in line for Hough
            ransac_threshold: RANSAC threshold for line fitting
            ransac_max_iters: Maximum RANSAC iterations
        """
        self.canny_low = canny_low
        self.canny_high = canny_high
        self.hough_threshold = hough_threshold
        self.hough_min_line_length = hough_min_line_length
        self.hough_max_line_gap = hough_max_line_gap
        self.ransac_threshold = ransac_threshold
        self.ransac_max_iters = ransac_max_iters
        
        # Store last computed homography for batch processing
        self.last_homography = None
    
    def detect_edges(self, image: np.ndarray) -> np.ndarray:
        """
        Detect edges in the image using Canny edge detector
        
        Args:
            image: Input image (BGR or grayscale)
        
        Returns:
            Binary edge map
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Canny edge detection
        edges = cv2.Canny(blurred, self.canny_low, self.canny_high)
        
        return edges
    
    def detect_lines(self, edges: np.ndarray) -> List[Tuple[float, float, float, float]]:
        """
        Detect lines using Hough Line Transform
        
        Args:
            edges: Binary edge map
        
        Returns:
            List of lines as (x1, y1, x2, y2) tuples
        """
        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi/180,
            threshold=self.hough_threshold,
            minLineLength=self.hough_min_line_length,
            maxLineGap=self.hough_max_line_gap
        )
        
        if lines is None:
            return []
        
        # Convert to list of tuples
        line_list = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            line_list.append((x1, y1, x2, y2))
        
        return line_list
    
    def estimate_horizon_line(self, 
                             lines: List[Tuple[float, float, float, float]],
                             image_shape: Tuple[int, int]) -> Optional[Tuple[float, float, float]]:
        """
        Estimate horizon line using RANSAC
        
        Args:
            lines: List of detected lines
            image_shape: (height, width) of the image
        
        Returns:
            Horizon line as (a, b, c) in ax + by + c = 0, or None if not found
        """
        if len(lines) < 2:
            return None
        
        # Convert lines to points for RANSAC
        # Use midpoints of lines in upper half of image
        h, w = image_shape
        points = []
        
        for x1, y1, x2, y2 in lines:
            # Only consider lines in upper half
            if y1 < h * 0.5 or y2 < h * 0.5:
                mid_x = (x1 + x2) / 2
                mid_y = (y1 + y2) / 2
                points.append([mid_x, mid_y])
        
        if len(points) < 2:
            return None
        
        points = np.array(points, dtype=np.float32)
        
        # Use RANSAC to fit a line
        try:
            # Fit line using least squares with RANSAC
            # We'll use a simple RANSAC implementation
            best_line = None
            best_inliers = 0
            
            for _ in range(self.ransac_max_iters):
                # Randomly sample 2 points
                sample_idx = np.random.choice(len(points), 2, replace=False)
                p1, p2 = points[sample_idx]
                
                # Compute line equation: ax + by + c = 0
                dx = p2[0] - p1[0]
                dy = p2[1] - p1[1]
                
                if abs(dx) < 1e-6 and abs(dy) < 1e-6:
                    continue
                
                # Normalize
                norm = np.sqrt(dx**2 + dy**2)
                a = -dy / norm
                b = dx / norm
                c = -(a * p1[0] + b * p1[1])
                
                # Count inliers
                distances = np.abs(a * points[:, 0] + b * points[:, 1] + c)
                inliers = np.sum(distances < self.ransac_threshold)
                
                if inliers > best_inliers:
                    best_inliers = inliers
                    best_line = (a, b, c)
            
            return best_line
            
        except Exception as e:
            warnings.warn(f"Error in RANSAC: {e}")
            return None
    
    def compute_homography(self, 
                          image_shape: Tuple[int, int],
                          horizon_line: Optional[Tuple[float, float, float]] = None) -> Optional[np.ndarray]:
        """
        Compute homography matrix to rectify the image
        
        Args:
            image_shape: (height, width) of the image
            horizon_line: Horizon line (a, b, c) or None for default
        
        Returns:
            3x3 homography matrix or None if computation fails
        """
        h, w = image_shape
        
        # Define source points (corners of the image)
        src_points = np.float32([
            [0, 0],      # Top-left
            [w, 0],      # Top-right
            [w, h],      # Bottom-right
            [0, h]       # Bottom-left
        ])
        
        # Define destination points for rectified image
        # If horizon line is available, use it to adjust perspective
        if horizon_line is not None:
            a, b, c = horizon_line
            
            # Find intersection of horizon line with image boundaries
            # Top edge: y = 0
            x_top_left = -c / a if abs(a) > 1e-6 else 0
            x_top_right = -(b * 0 + c) / a if abs(a) > 1e-6 else w
            
            # Adjust top points based on horizon
            top_y = max(0, min(h // 4, h // 3))  # Keep horizon in upper portion
            src_points[0][1] = top_y
            src_points[1][1] = top_y
        
        # Destination points: create a more "straight" perspective
        # Move top points up slightly to create bird's-eye-like view
        dst_points = np.float32([
            [0, 0],                    # Top-left (moved up)
            [w, 0],                    # Top-right (moved up)
            [w * 0.95, h],             # Bottom-right (slightly inward)
            [w * 0.05, h]              # Bottom-left (slightly inward)
        ])
        
        # Compute homography
        try:
            H = cv2.getPerspectiveTransform(src_points, dst_points)
            return H
        except Exception as e:
            warnings.warn(f"Error computing homography: {e}")
            return None
    
    def rectify_image(self, 
                     image: np.ndarray,
                     homography: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Rectify an image using homography transformation
        
        Args:
            image: Input image to rectify
            homography: Pre-computed homography matrix (if None, will compute)
        
        Returns:
            Tuple of (rectified_image, homography_matrix)
        """
        h, w = image.shape[:2]
        
        if homography is None:
            # Detect edges
            edges = self.detect_edges(image)
            
            # Detect lines
            lines = self.detect_lines(edges)
            
            # Estimate horizon
            horizon = self.estimate_horizon_line(lines, (h, w))
            
            # Compute homography
            homography = self.compute_homography((h, w), horizon)
            
            if homography is None:
                # Return original image if homography computation fails
                return image, np.eye(3, dtype=np.float32)
        
        # Store for potential reuse
        self.last_homography = homography
        
        # Apply homography
        rectified = cv2.warpPerspective(
            image,
            homography,
            (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0
        )
        
        return rectified, homography
    
    def rectify_mask(self, 
                    mask: np.ndarray,
                    homography: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Rectify a binary mask using the same homography
        
        Args:
            mask: Binary mask to rectify
            homography: Homography matrix (uses last computed if None)
        
        Returns:
            Rectified mask
        """
        if homography is None:
            if self.last_homography is not None:
                homography = self.last_homography
            else:
                return mask
        
        h, w = mask.shape[:2]
        rectified = cv2.warpPerspective(
            mask,
            homography,
            (w, h),
            flags=cv2.INTER_NEAREST,  # Use nearest neighbor for masks
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0
        )
        
        return rectified


def rectify_frame(image: np.ndarray, 
                 rectifier: Optional[HomographyRectifier] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convenience function to rectify a single frame
    
    Args:
        image: Input image (BGR format)
        rectifier: HomographyRectifier instance (creates new one if None)
    
    Returns:
        Tuple of (rectified_image, homography_matrix)
    """
    if rectifier is None:
        rectifier = HomographyRectifier()
    
    return rectifier.rectify_image(image)

