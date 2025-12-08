import os
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
from torchvision import transforms

class SequenceDataset(Dataset):
    """
    Devuelve 3 frames consecutivos: (I_{t-2}, I_{t-1}, I_t)
    """
    def __init__(self, image_paths, mask_paths, sequence_len=3, target_size=(512,256)):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.seq = sequence_len
        self.size = target_size

    def __len__(self):
        return len(self.image_paths)

    def load_img(self, p):
        # Handle truncated/corrupted images or missing files
        import os
        import cv2
        import numpy as np
        
        # Check if file exists first
        if not os.path.exists(p):
            print(f"Warning: Image file not found: {p}, using empty image")
            # Create empty image with target size
            img = Image.new('RGB', self.size, color=(0, 0, 0))
            img_tensor = TF.to_tensor(img)  # [0, 1] range
            normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            img_tensor = normalize(img_tensor)
            return img_tensor
        
        img = None
        max_retries = 3
        for attempt in range(max_retries):
            try:
                img = Image.open(p)
                # Verify image is not truncated
                img.load()
                img = img.convert("RGB")
                break
            except (OSError, IOError, FileNotFoundError) as e:
                if attempt < max_retries - 1:
                    continue
                else:
                    # Fallback to cv2
                    cv_img = cv2.imread(p, cv2.IMREAD_COLOR)
                    if cv_img is not None and cv_img.size > 0:
                        img = Image.fromarray(cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB))
                    else:
                        # Image is completely corrupted or missing, create empty image
                        print(f"Warning: Image {p} is corrupted or cannot be read, using empty image")
                        # Create empty image with target size
                        img = Image.new('RGB', self.size, color=(0, 0, 0))
        
        img = img.resize(self.size)
        # Convert to tensor and normalize with ImageNet stats (same as non-temporal mode)
        img_tensor = TF.to_tensor(img)  # [0, 1] range
        # Apply ImageNet normalization: mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        img_tensor = normalize(img_tensor)
        return img_tensor
    
    def load_mask(self, p):
        """
        Load binary mask - handles both grayscale and RGB mask images
        Similar to how TusimpleSet processes masks
        """
        import cv2
        import numpy as np
        import os
        
        # Check if file exists first
        if not os.path.exists(p):
            print(f"Warning: Mask file not found: {p}, using empty mask")
            # Return empty mask with target size
            label_binary = np.zeros([self.size[1], self.size[0]], dtype=np.float32)
            mask_tensor = torch.from_numpy(label_binary).unsqueeze(0)
            return mask_tensor
        
        # Try loading as grayscale first (more efficient for binary masks)
        label_img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        if label_img is not None and label_img.size > 0:
            # Resize to target size
            label_img = cv2.resize(label_img, self.size)
            # Binarize: values > 127 (midpoint) become 1, else 0
            label_binary = (label_img > 127).astype(np.float32)
            # Convert to tensor [1, H, W] to match expected format
            mask_tensor = torch.from_numpy(label_binary).unsqueeze(0)
            return mask_tensor
        
        # Fallback: Try loading as RGB
        label_img = cv2.imread(p, cv2.IMREAD_COLOR)
        if label_img is not None and label_img.size > 0:
            # Resize to target size
            label_img = cv2.resize(label_img, self.size)
            # Convert to binary mask (same logic as TusimpleSet)
            # Pixels that are not [0, 0, 0] (black) become 1, else 0
            label_binary = np.zeros([label_img.shape[0], label_img.shape[1]], dtype=np.float32)
            # Check if any channel is non-zero (more robust than checking all channels)
            mask = np.where(np.any(label_img != [0, 0, 0], axis=2))
            label_binary[mask] = 1.0
            # Convert to tensor [1, H, W] to match expected format
            mask_tensor = torch.from_numpy(label_binary).unsqueeze(0)
            return mask_tensor
        
        # Final fallback: Try PIL, or create empty mask if file is corrupted
        try:
            m = Image.open(p).convert("L")
            m = m.resize(self.size)
            mask_tensor = TF.to_tensor(m)
            # Binarize: values > 0.5 become 1, else 0
            mask_tensor = (mask_tensor > 0.5).float()
            return mask_tensor
        except (OSError, IOError, FileNotFoundError) as e:
            print(f"Warning: Could not load mask {p}: {e}, using empty mask")
            # Return empty mask with target size
            label_binary = np.zeros([self.size[1], self.size[0]], dtype=np.float32)
            mask_tensor = torch.from_numpy(label_binary).unsqueeze(0)
            return mask_tensor

    def __getitem__(self, idx):
        start = max(0, idx - self.seq + 1)

        frames = []
        for i in range(start, idx + 1):
            frames.append(self.load_img(self.image_paths[i]))

        while len(frames) < self.seq:
            frames.insert(0, frames[0])

        stacked = torch.cat(frames, dim=0)  # 3*3=9 channels

        mask = self.load_mask(self.mask_paths[idx])
        
        # Debug first sample
        if idx == 0:
            print(f"Debug SequenceDataset - mask_path: {self.mask_paths[idx]}")
            print(f"Debug SequenceDataset - mask shape: {mask.shape}")
            print(f"Debug SequenceDataset - mask min: {mask.min()}, max: {mask.max()}, sum: {mask.sum()}")
            print(f"Debug SequenceDataset - mask unique values: {torch.unique(mask)}")
        
        return stacked, mask
