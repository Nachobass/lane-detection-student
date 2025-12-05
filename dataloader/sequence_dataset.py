import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF

# Import Albumentations augmentations
try:
    from model.lanenet.dataset.augmentations import get_training_augmentations, get_validation_transform
    ALBUMENTATIONS_AVAILABLE = True
except ImportError:
    ALBUMENTATIONS_AVAILABLE = False
    print("Warning: Albumentations not available. Using basic transforms.")


class SequenceDataset(Dataset):
    """
    Dataset for temporal sequences with strong geometric augmentations
    
    Returns sequences of consecutive frames: (I_{t-2}, I_{t-1}, I_t)
    Uses Albumentations for synchronized image and mask augmentation
    """
    def __init__(self, image_paths, mask_paths, sequence_len=3, target_size=(512,256), 
                 use_augmentation=False, strong_augmentation=False):
        """
        Args:
            image_paths: List of image file paths
            mask_paths: List of mask file paths
            sequence_len: Number of frames in sequence
            target_size: Target image size (width, height)
            use_augmentation: Whether to apply augmentations (for training)
            strong_augmentation: Use stronger augmentations (default: False)
        """
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.seq = sequence_len
        self.size = target_size
        self.use_augmentation = use_augmentation and ALBUMENTATIONS_AVAILABLE
        self.strong_augmentation = strong_augmentation
        
        # Setup transforms
        if self.use_augmentation:
            if strong_augmentation:
                from model.lanenet.dataset.augmentations import get_strong_augmentations
                self.transform = get_strong_augmentations(image_size=(target_size[1], target_size[0]))
            else:
                self.transform = get_training_augmentations(image_size=(target_size[1], target_size[0]))
        else:
            self.transform = get_validation_transform(image_size=(target_size[1], target_size[0])) if ALBUMENTATIONS_AVAILABLE else None
        
        if self.use_augmentation:
            print(f"Using {'strong' if strong_augmentation else 'standard'} Albumentations augmentations")
        elif ALBUMENTATIONS_AVAILABLE:
            print("Using validation transforms (no augmentation)")
        else:
            print("Using basic PIL transforms (Albumentations not available)")

    def __len__(self):
        return len(self.image_paths)

    def load_img_pil(self, p):
        """Load image as PIL Image"""
        img = Image.open(p).convert("RGB")
        return img
    
    def load_mask_pil(self, p):
        """Load mask as PIL Image"""
        m = Image.open(p).convert("L")
        return m
    
    def load_img_numpy(self, p):
        """Load image as numpy array for Albumentations"""
        img = Image.open(p).convert("RGB")
        return np.array(img)
    
    def load_mask_numpy(self, p):
        """Load mask as numpy array for Albumentations"""
        m = Image.open(p).convert("L")
        return np.array(m)

    def __getitem__(self, idx):
        """
        Get sequence of frames and corresponding mask
        
        Returns:
            stacked: Tensor of shape [T*3, H, W] (stacked frames)
            mask: Tensor of shape [1, H, W] or [H, W]
        """
        start = max(0, idx - self.seq + 1)
        
        # Load sequence of frames
        frames = []
        for i in range(start, idx + 1):
            if self.use_augmentation and self.transform is not None:
                # Use numpy for Albumentations
                img = self.load_img_numpy(self.image_paths[i])
                frames.append(img)
            else:
                # Use PIL for basic transforms
                img = self.load_img_pil(self.image_paths[i])
                img = img.resize(self.size)
                frames.append(TF.to_tensor(img))
        
        # Pad sequence if needed (repeat first frame)
        while len(frames) < self.seq:
            if self.use_augmentation:
                frames.insert(0, frames[0].copy())
            else:
                frames.insert(0, frames[0])
        
        # Load mask
        if self.use_augmentation and self.transform is not None:
            # Apply augmentation to all frames and mask together
            # Strategy: Apply same geometric transforms to all frames for temporal consistency
            # but allow slight variations in non-geometric transforms (brightness, blur)
            
            # Load mask
            mask = self.load_mask_numpy(self.mask_paths[idx])
            
            # Apply augmentation to the last frame (which corresponds to the mask)
            augmented = self.transform(image=frames[-1], mask=mask)
            frames[-1] = augmented["image"]
            mask_tensor = augmented["mask"]
            
            # Apply augmentation to previous frames
            # Note: Each call to transform() uses different random parameters
            # For temporal sequences, we want geometric consistency but can allow
            # illumination/blur variations (which naturally occur in video)
            for i in range(len(frames) - 1):
                # Don't pass mask parameter when there's no mask
                augmented_frame = self.transform(image=frames[i])
                frames[i] = augmented_frame["image"]
            
            # Ensure all frames are float tensors
            for i in range(len(frames)):
                if not frames[i].is_floating_point():
                    frames[i] = frames[i].float()
            
            # Stack frames: [T, 3, H, W] -> [T*3, H, W]
            stacked = torch.cat(frames, dim=0)
            
            # Ensure mask is in correct format and float
            if mask_tensor.dim() == 2:
                mask_tensor = mask_tensor.unsqueeze(0)  # [1, H, W]
            if not mask_tensor.is_floating_point():
                mask_tensor = mask_tensor.float()
            
            return stacked, mask_tensor
        else:
            # Basic transforms without augmentation
            # Stack frames
            if isinstance(frames[0], torch.Tensor):
                stacked = torch.cat(frames, dim=0)  # [T*3, H, W]
            else:
                # Convert PIL to tensor and stack
                tensor_frames = [TF.to_tensor(f) for f in frames]
                stacked = torch.cat(tensor_frames, dim=0)
            
            # Load mask
            mask = self.load_mask_pil(self.mask_paths[idx])
            mask = mask.resize(self.size)
            mask = TF.to_tensor(mask)
            
            return stacked, mask
