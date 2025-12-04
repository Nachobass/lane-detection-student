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
        m = Image.open(p).convert("L")
        m = m.resize(self.size)
        return TF.to_tensor(m)

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
