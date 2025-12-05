import os
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF

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
        img = Image.open(p).convert("RGB")
        img = img.resize(self.size)
        return TF.to_tensor(img)
    
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
        start = max(0, idx - self.seq + 1)

        frames = []
        for i in range(start, idx + 1):
            frames.append(self.load_img(self.image_paths[i]))

        while len(frames) < self.seq:
            frames.insert(0, frames[0])

        stacked = torch.cat(frames, dim=0)  # 3*3=9 channels

        mask = self.load_mask(self.mask_paths[idx])
        return stacked, mask
