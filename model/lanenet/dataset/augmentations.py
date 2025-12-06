# coding: utf-8
"""
Strong geometric augmentations for lane detection using Albumentations
Designed for urban lane detection with temporal sequences

Uses additional_targets to ensure the same geometric transformation
is applied to all frames in a sequence simultaneously.
"""
import albumentations as A
from albumentations.pytorch import ToTensorV2


def get_training_augmentations(image_size=(512, 256), sequence_length=3):
    """
    Strong geometric augmentations for training with temporal sequence support
    
    Uses additional_targets to apply the same geometric transformation
    to all frames in the sequence simultaneously, preserving temporal consistency.
    
    Includes:
    - Perspective transformations (simulates viewpoint changes)
    - ShiftScaleRotate (camera roll, pitch, zoom)
    - Brightness/Contrast (handles shadows and overexposure)
    - GaussianBlur (simulates camera shake/low quality)
    - CoarseDropout (Cutout - forces model to complete lanes)
    
    Args:
        image_size: Tuple of (height, width)
        sequence_length: Number of frames in sequence (default: 3)
    
    Returns:
        Albumentations Compose transform with additional_targets configured
    """
    H, W = image_size
    
    # Define additional targets for previous frames in sequence
    # For sequence_length=3, we have: image (current), image0 (t-2), image1 (t-1)
    additional_targets = {}
    for i in range(sequence_length - 1):
        additional_targets[f'image{i}'] = 'image'
    
    return A.Compose([
        A.Resize(height=H, width=W),
        
        # --- GEOMETRIC AUGMENTATIONS ---
        # ShiftScaleRotate: simulates camera roll, pitch, and zoom changes
        # Same transformation applied to all frames via additional_targets
        A.ShiftScaleRotate(
            shift_limit=0.05,      # 5% shift
            scale_limit=0.1,        # 10% scale (zoom in/out)
            rotate_limit=10,       # ±10 degrees rotation
            border_mode=0,          # Constant padding (black)
            p=0.7                  # 70% probability
        ),
        
        # Perspective: simulates viewpoint/perspective changes
        # Same transformation applied to all frames
        A.Perspective(
            scale=(0.03, 0.05),    # 3-5% perspective distortion
            keep_size=True,         # Keep original size
            p=0.5                  # 50% probability
        ),
        
        # --- ILLUMINATION AUGMENTATIONS ---
        # RandomBrightnessContrast: handles shadows and overexposure
        # Applied independently to each frame (natural video variation)
        A.RandomBrightnessContrast(
            brightness_limit=0.3,  # ±30% brightness
            contrast_limit=0.3,     # ±30% contrast
            p=0.7                  # 70% probability
        ),
        
        # --- BLUR AUGMENTATIONS ---
        # GaussianBlur: simulates camera shake or low quality frames
        # Applied independently to each frame
        A.GaussianBlur(
            blur_limit=(3, 7),      # Kernel size 3x3 to 7x7
            p=0.3                  # 30% probability
        ),
        
        # --- OCCLUSION AUGMENTATIONS ---
        # CoarseDropout (Cutout): removes parts of image
        # Same dropout pattern applied to all frames (consistent occlusion)
        A.CoarseDropout(
            max_holes=8,           # Maximum 8 holes
            max_height=32,          # Max hole height
            max_width=32,           # Max hole width
            fill_value=0,           # Fill with black
            p=0.4                  # 40% probability
        ),
        
        # Convert to tensor (normalized to [0, 1])
        ToTensorV2()
    ], additional_targets=additional_targets)


def get_validation_transform(image_size=(512, 256), sequence_length=3):
    """
    Simple validation transform (no augmentations)
    
    Args:
        image_size: Tuple of (height, width)
        sequence_length: Number of frames in sequence (default: 3)
    
    Returns:
        Albumentations Compose transform with additional_targets configured
    """
    H, W = image_size
    
    # Define additional targets for previous frames
    additional_targets = {}
    for i in range(sequence_length - 1):
        additional_targets[f'image{i}'] = 'image'
    
    return A.Compose([
        A.Resize(height=H, width=W),
        ToTensorV2()
    ], additional_targets=additional_targets)


def get_strong_augmentations(image_size=(512, 256), sequence_length=3):
    """
    Even stronger augmentations for challenging scenarios
    
    Use this if you need more aggressive augmentation.
    Same geometric transformations are applied to all frames via additional_targets.
    
    Args:
        image_size: Tuple of (height, width)
        sequence_length: Number of frames in sequence (default: 3)
    
    Returns:
        Albumentations Compose transform with additional_targets configured
    """
    H, W = image_size
    
    # Define additional targets for previous frames
    additional_targets = {}
    for i in range(sequence_length - 1):
        additional_targets[f'image{i}'] = 'image'
    
    return A.Compose([
        A.Resize(height=H, width=W),
        
        # Stronger geometric augmentations (applied to all frames)
        A.ShiftScaleRotate(
            shift_limit=0.1,
            scale_limit=0.2,
            rotate_limit=15,
            border_mode=0,
            p=0.8
        ),
        
        A.Perspective(
            scale=(0.05, 0.1),
            keep_size=True,
            p=0.6
        ),
        
        # Additional geometric: ElasticTransform (applied to all frames)
        A.ElasticTransform(
            alpha=50,
            sigma=5,
            alpha_affine=50,
            p=0.3
        ),
        
        # Illumination (applied independently per frame)
        A.RandomBrightnessContrast(
            brightness_limit=0.4,
            contrast_limit=0.4,
            p=0.8
        ),
        
        # Color augmentations (applied independently per frame)
        A.HueSaturationValue(
            hue_shift_limit=10,
            sat_shift_limit=20,
            val_shift_limit=20,
            p=0.5
        ),
        
        # Blur (applied independently per frame)
        A.GaussianBlur(blur_limit=(3, 7), p=0.4),
        
        # Occlusion (same pattern applied to all frames)
        A.CoarseDropout(
            max_holes=12,
            max_height=48,
            max_width=48,
            fill_value=0,
            p=0.5
        ),
        
        ToTensorV2()
    ], additional_targets=additional_targets)


