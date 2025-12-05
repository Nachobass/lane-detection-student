import time
import os
import sys

import torch
from dataloader.data_loaders import TusimpleSet
from dataloader.sequence_dataset import SequenceDataset
from dataloader.transformers import Rescale
from model.lanenet.LaneNet import LaneNet
from torch.utils.data import DataLoader, dataloader
from torch.autograd import Variable

from torchvision import transforms

from model.utils.cli_helper_eval import parse_args
from model.eval_function import Eval_Score
from model.lanenet.train_lanenet import reshape_sequence_input

import numpy as np
from PIL import Image
import pandas as pd
import cv2

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def evaluation():
    args = parse_args()
    resize_height = args.height
    resize_width = args.width

    data_transform = transforms.Compose([
        transforms.Resize((resize_height,  resize_width)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    target_transforms = transforms.Compose([
        Rescale((resize_width, resize_height)),
    ])

    dataset_file = os.path.join(args.dataset, 'test.txt')
    
    # Check if temporal mode is enabled
    if args.use_temporal:
        # Load image and mask paths from test.txt for SequenceDataset
        test_image_paths = []
        test_mask_paths = []
        with open(dataset_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    test_image_paths.append(parts[0])
                    test_mask_paths.append(parts[1])  # Using binary mask path
        
        # Create SequenceDataset for temporal evaluation
        Eval_Dataset = SequenceDataset(
            test_image_paths,
            test_mask_paths,
            sequence_len=args.sequence_length,
            target_size=(resize_width, resize_height)
        )
        print(f"Using temporal evaluation with sequence_length={args.sequence_length}")
        print(f"Test samples: {len(Eval_Dataset)}")
    else:
        # Standard non-temporal evaluation
        Eval_Dataset = TusimpleSet(dataset_file, transform=data_transform, target_transform=target_transforms)
        print("Using standard (non-temporal) evaluation")
    
    eval_dataloader = DataLoader(Eval_Dataset, batch_size=1, shuffle=False)

    model_path = args.model
    model = LaneNet(arch=args.model_type, use_temporal=args.use_temporal, sequence_length=args.sequence_length)
    state_dict = torch.load(model_path, map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.eval()
    model.to(DEVICE)
    
    print(f"Model loaded from: {model_path}")
    print(f"Model type: {args.model_type}, Temporal: {args.use_temporal}")

    iou, dice = 0, 0
    with torch.no_grad():
        for batch_idx, batch_data in enumerate(eval_dataloader):
            if args.use_temporal:
                # Temporal mode: batch_data is (images, masks)
                x, masks = batch_data
                # Reshape if needed: [B, T*3, H, W] -> [B, T, 3, H, W]
                if x.dim() == 4 and x.shape[1] == args.sequence_length * 3:
                    x = reshape_sequence_input(x, args.sequence_length)
                x = x.to(DEVICE)
                target = masks.to(DEVICE)
                
                # Handle mask shape
                if target.dim() == 4:
                    target = target.squeeze(1)  # [B, H, W]
            else:
                # Non-temporal mode: batch_data is (images, binary_masks, instance_masks)
                x, target, _ = batch_data
                x = x.to(DEVICE)
                target = target.to(DEVICE)
            
            y = model(x)
            y_pred = torch.squeeze(y['binary_seg_pred'].to('cpu')).numpy()
            y_true = torch.squeeze(target.to('cpu')).numpy()
            Score = Eval_Score(y_pred, y_true)
            dice += Score.Dice()
            iou += Score.IoU()
    
    final_iou = iou / len(eval_dataloader.dataset)
    final_dice = dice / len(eval_dataloader.dataset)
    
    print('='*60)
    print('EVALUATION RESULTS:')
    print('='*60)
    print(f'Final_IoU: {final_iou:.4f} ({final_iou*100:.2f}%)')
    print(f'Final_F1 (Dice): {final_dice:.4f} ({final_dice*100:.2f}%)')
    print('='*60)


if __name__ == "__main__":
    evaluation()