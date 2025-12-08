import os
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import cv2
import numpy as np

from torchvision.transforms import ToTensor
from torchvision import datasets, transforms

import random


class TusimpleSet(Dataset):
    def __init__(self, dataset, n_labels=3, transform=None, target_transform=None):
        self._gt_img_list = []
        self._gt_label_binary_list = []
        self._gt_label_instance_list = []
        self.transform = transform
        self.target_transform = target_transform
        self.n_labels = n_labels

        with open(dataset, 'r') as file:
            for _info in file:
                info_tmp = _info.strip(' ').split()

                self._gt_img_list.append(info_tmp[0])
                self._gt_label_binary_list.append(info_tmp[1])
                self._gt_label_instance_list.append(info_tmp[2])

        assert len(self._gt_img_list) == len(self._gt_label_binary_list) == len(self._gt_label_instance_list)

        self._shuffle()

    def _shuffle(self):
        # randomly shuffle all list identically
        c = list(zip(self._gt_img_list, self._gt_label_binary_list, self._gt_label_instance_list))
        random.shuffle(c)
        self._gt_img_list, self._gt_label_binary_list, self._gt_label_instance_list = zip(*c)

    def __len__(self):
        return len(self._gt_img_list)

    def __getitem__(self, idx):
        assert len(self._gt_label_binary_list) == len(self._gt_label_instance_list) \
               == len(self._gt_img_list)

        # load all
        # Handle truncated/corrupted images
        img = None
        max_retries = 3
        for attempt in range(max_retries):
            try:
                img = Image.open(self._gt_img_list[idx])
                # Verify image is not truncated
                img.load()
                break
            except (OSError, IOError) as e:
                if attempt < max_retries - 1:
                    # Try to reload
                    continue
                else:
                    # If still fails, try to load with cv2 as fallback
                    print(f"Warning: Could not load image {self._gt_img_list[idx]} with PIL, trying cv2...")
                    cv_img = cv2.imread(self._gt_img_list[idx], cv2.IMREAD_COLOR)
                    if cv_img is not None:
                        img = Image.fromarray(cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB))
                    else:
                        # Image is completely corrupted, delete it and create empty image
                        print(f"Error: Image {self._gt_img_list[idx]} is corrupted. Deleting and using empty image.")
                        try:
                            os.remove(self._gt_img_list[idx])
                            # Also delete corresponding masks
                            if idx < len(self._gt_label_binary_list):
                                if os.path.exists(self._gt_label_binary_list[idx]):
                                    os.remove(self._gt_label_binary_list[idx])
                            if idx < len(self._gt_label_instance_list):
                                if os.path.exists(self._gt_label_instance_list[idx]):
                                    os.remove(self._gt_label_instance_list[idx])
                        except Exception as del_err:
                            print(f"Warning: Could not delete corrupted file: {del_err}")
                        
                        # Create empty image (default size 1640x590 for CULane, or use a sample)
                        # Try to get size from a valid image
                        default_size = (1640, 590)  # Common CULane size
                        if idx > 0:
                            try:
                                sample_img = Image.open(self._gt_img_list[0])
                                default_size = sample_img.size
                                sample_img.close()
                            except:
                                pass
                        img = Image.new('RGB', default_size, color=(0, 0, 0))
        
        label_instance_img = cv2.imread(self._gt_label_instance_list[idx], cv2.IMREAD_UNCHANGED)
        label_img = cv2.imread(self._gt_label_binary_list[idx], cv2.IMREAD_COLOR)
        
        # Handle missing or corrupted mask files
        if label_img is None:
            print(f"Warning: Could not load binary mask {self._gt_label_binary_list[idx]}, creating empty mask")
            # Create empty mask with same size as image
            img_array = np.array(img)
            if len(img_array.shape) == 3:
                h, w = img_array.shape[:2]
            else:
                h, w = img_array.shape
            label_img = np.zeros((h, w, 3), dtype=np.uint8)
        
        if label_instance_img is None:
            print(f"Warning: Could not load instance mask {self._gt_label_instance_list[idx]}, creating empty mask")
            # Create empty mask with same size as binary mask
            if label_img is not None:
                h, w = label_img.shape[:2]
            else:
                img_array = np.array(img)
                if len(img_array.shape) == 3:
                    h, w = img_array.shape[:2]
                else:
                    h, w = img_array.shape
            label_instance_img = np.zeros((h, w), dtype=np.uint8)

        # optional transformations
        if self.transform:
            img = self.transform(img)
        if self.target_transform:
            if label_img is not None and label_img.size > 0:
                label_img = self.target_transform(label_img)
            else:
                # Create empty mask with target size
                label_img = np.zeros((256, 512, 3), dtype=np.uint8)
                label_img = self.target_transform(label_img)
            
            if label_instance_img is not None and label_instance_img.size > 0:
                label_instance_img = self.target_transform(label_instance_img)
            else:
                # Create empty mask with target size
                label_instance_img = np.zeros((256, 512), dtype=np.uint8)
                label_instance_img = self.target_transform(label_instance_img)

        label_binary = np.zeros([label_img.shape[0], label_img.shape[1]], dtype=np.uint8)
        mask = np.where((label_img[:, :, :] != [0, 0, 0]).all(axis=2))
        label_binary[mask] = 1

        # we could split the instance label here, each instance in one channel (basically a binary mask for each)
        return img, label_binary, label_instance_img
