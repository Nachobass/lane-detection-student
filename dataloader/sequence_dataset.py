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
        
        # Setup transforms with sequence_length support
        if self.use_augmentation:
            if strong_augmentation:
                from model.lanenet.dataset.augmentations import get_strong_augmentations
                self.transform = get_strong_augmentations(
                    image_size=(target_size[1], target_size[0]),
                    sequence_length=sequence_len
                )
            else:
                self.transform = get_training_augmentations(
                    image_size=(target_size[1], target_size[0]),
                    sequence_length=sequence_len
                )
        else:
            self.transform = get_validation_transform(
                image_size=(target_size[1], target_size[0]),
                sequence_length=sequence_len
            ) if ALBUMENTATIONS_AVAILABLE else None
        
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
    
    def load_img_numpy(self, p):
        """Load image as numpy array for Albumentations"""
        img = Image.open(p).convert("RGB")
        return np.array(img)
    
    def load_mask_numpy(self, p):
        """Load mask as numpy array for Albumentations"""
        import cv2
        # Try loading as RGB first (like TusimpleSet does)
        label_img = cv2.imread(p, cv2.IMREAD_COLOR)
        if label_img is None:
            # Fallback to PIL if cv2 fails
            m = Image.open(p).convert("L")
            return np.array(m)
        
        # Convert to binary mask (same logic as TusimpleSet)
        # Pixels that are not [0, 0, 0] (black) become 1, else 0
        label_binary = np.zeros([label_img.shape[0], label_img.shape[1]], dtype=np.float32)
        mask = np.where((label_img[:, :, :] != [0, 0, 0]).all(axis=2))
        label_binary[mask] = 1.0
        return label_binary.astype(np.uint8) * 255  # Convert to uint8 for Albumentations
    
    def load_mask_pil(self, p):
        """Load mask as PIL Image"""
        m = Image.open(p).convert("L")
        return m
    
    def load_mask(self, p):
        """
        Load binary mask - handles both grayscale and RGB mask images
        Similar to how TusimpleSet processes masks
        """
        import cv2
        import numpy as np
        
        # Try loading as RGB first (like TusimpleSet does)
        label_img = cv2.imread(p, cv2.IMREAD_COLOR)
        if label_img is None:
            # Fallback to PIL if cv2 fails
            m = Image.open(p).convert("L")
            m = m.resize(self.size)
            mask_tensor = TF.to_tensor(m)
            # Binarize: values > 0.5 become 1, else 0
            mask_tensor = (mask_tensor > 0.5).float()
            return mask_tensor
        
        # Resize to target size
        label_img = cv2.resize(label_img, self.size)
        
        # Convert to binary mask (same logic as TusimpleSet)
        # Pixels that are not [0, 0, 0] (black) become 1, else 0
        label_binary = np.zeros([label_img.shape[0], label_img.shape[1]], dtype=np.float32)
        mask = np.where((label_img[:, :, :] != [0, 0, 0]).all(axis=2))
        label_binary[mask] = 1.0
        
        # Convert to tensor [1, H, W] to match expected format
        mask_tensor = torch.from_numpy(label_binary).unsqueeze(0)
        return mask_tensor

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
            # Apply augmentation to ALL frames simultaneously using additional_targets
            # This ensures the same geometric transformation (rotation, perspective, etc.)
            # is applied to all frames, preserving temporal consistency
            
            # Load mask
            mask = self.load_mask_numpy(self.mask_paths[idx])
            
            # Prepare input dictionary for Albumentations with additional_targets
            # Format: {'image': current_frame, 'image0': frame_t-2, 'image1': frame_t-1, 'mask': mask}
            transform_input = {
                'image': frames[-1],  # Current frame (t)
                'mask': mask
            }
            
            # Add previous frames as additional targets
            # For sequence_length=3: frames[-1] is t, frames[-2] is t-1, frames[-3] is t-2
            # We map: image0 -> t-2, image1 -> t-1 (if sequence_length=3)
            for i in range(len(frames) - 1):
                transform_input[f'image{i}'] = frames[i]
            
            # Apply transformation to all frames simultaneously
            # Geometric transforms (ShiftScaleRotate, Perspective) will be applied
            # identically to all frames via additional_targets
            augmented = self.transform(**transform_input)
            
            # Extract augmented frames
            frames[-1] = augmented['image']  # Current frame
            for i in range(len(frames) - 1):
                frames[i] = augmented[f'image{i}']  # Previous frames
            
            # Extract mask
            mask_tensor = augmented['mask']
            
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
