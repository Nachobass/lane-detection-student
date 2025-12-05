# coding: utf-8
"""
Strong geometric augmentations for lane detection using Albumentations
Designed for urban lane detection with temporal sequences
"""
import albumentations as A
from albumentations.pytorch import ToTensorV2


def get_training_augmentations(image_size=(512, 256)):
    """
    Strong geometric augmentations for training
    
    Includes:
    - Perspective transformations (simulates viewpoint changes)
    - ShiftScaleRotate (camera roll, pitch, zoom)
    - Brightness/Contrast (handles shadows and overexposure)
    - GaussianBlur (simulates camera shake/low quality)
    - CoarseDropout (Cutout - forces model to complete lanes)
    
    Args:
        image_size: Tuple of (height, width)
    
    Returns:
        Albumentations Compose transform
    """
    H, W = image_size
    
    return A.Compose([
        A.Resize(height=H, width=W),
        
        # --- GEOMETRIC AUGMENTATIONS ---
        # ShiftScaleRotate: simulates camera roll, pitch, and zoom changes
        A.ShiftScaleRotate(
            shift_limit=0.05,      # 5% shift
            scale_limit=0.1,        # 10% scale (zoom in/out)
            rotate_limit=10,       # ±10 degrees rotation
            border_mode=0,          # Constant padding (black)
            p=0.7                  # 70% probability
        ),
        
        # Perspective: simulates viewpoint/perspective changes
        A.Perspective(
            scale=(0.03, 0.05),    # 3-5% perspective distortion
            keep_size=True,         # Keep original size
            p=0.5                  # 50% probability
        ),
        
        # --- ILLUMINATION AUGMENTATIONS ---
        # RandomBrightnessContrast: handles shadows and overexposure
        A.RandomBrightnessContrast(
            brightness_limit=0.3,  # ±30% brightness
            contrast_limit=0.3,     # ±30% contrast
            p=0.7                  # 70% probability
        ),
        
        # --- BLUR AUGMENTATIONS ---
        # GaussianBlur: simulates camera shake or low quality frames
        A.GaussianBlur(
            blur_limit=(3, 7),      # Kernel size 3x3 to 7x7
            p=0.3                  # 30% probability
        ),
        
        # --- OCCLUSION AUGMENTATIONS ---
        # CoarseDropout (Cutout): removes parts of image
        # Forces model to "remember" and complete lanes (especially useful for ConvLSTM)
        A.CoarseDropout(
            max_holes=8,           # Maximum 8 holes
            max_height=32,          # Max hole height
            max_width=32,           # Max hole width
            fill_value=0,           # Fill with black
            p=0.4                  # 40% probability
        ),
        
        # Convert to tensor (normalized to [0, 1])
        ToTensorV2()
    ])


def get_validation_transform(image_size=(512, 256)):
    """
    Simple validation transform (no augmentations)
    
    Args:
        image_size: Tuple of (height, width)
    
    Returns:
        Albumentations Compose transform
    """
    H, W = image_size
    
    return A.Compose([
        A.Resize(height=H, width=W),
        ToTensorV2()
    ])


def get_strong_augmentations(image_size=(512, 256)):
    """
    Even stronger augmentations for challenging scenarios
    
    Use this if you need more aggressive augmentation
    
    Args:
        image_size: Tuple of (height, width)
    
    Returns:
        Albumentations Compose transform
    """
    H, W = image_size
    
    return A.Compose([
        A.Resize(height=H, width=W),
        
        # Stronger geometric augmentations
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
        
        # Additional geometric: ElasticTransform
        A.ElasticTransform(
            alpha=50,
            sigma=5,
            alpha_affine=50,
            p=0.3
        ),
        
        # Illumination
        A.RandomBrightnessContrast(
            brightness_limit=0.4,
            contrast_limit=0.4,
            p=0.8
        ),
        
        # Color augmentations
        A.HueSaturationValue(
            hue_shift_limit=10,
            sat_shift_limit=20,
            val_shift_limit=20,
            p=0.5
        ),
        
        # Blur
        A.GaussianBlur(blur_limit=(3, 7), p=0.4),
        
        # Occlusion
        A.CoarseDropout(
            max_holes=12,
            max_height=48,
            max_width=48,
            fill_value=0,
            p=0.5
        ),
        
        ToTensorV2()
    ])


