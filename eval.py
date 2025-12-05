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
    
    # Create output directory if saving images
    save_dir = args.save
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        print(f"Output images will be saved to: {save_dir}")
    
    # Limit number of images to save (optional, set to None to save all)
    max_images_to_save = 10  # Save first 10 images as examples

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
            
            # Process ground truth mask - ensure it's binary
            y_true = target.to('cpu')
            if y_true.dim() > 2:
                y_true = torch.squeeze(y_true)
            
            # Convert to numpy and ensure binary (0 or 1)
            # SequenceDataset loads masks as normalized [0, 1], need to binarize
            y_true_np = y_true.numpy()
            
            # Binarize: values > 0.5 become 1, else 0
            # This handles both normalized masks [0, 1] and already binary masks
            y_true_np = (y_true_np > 0.5).astype(np.float32)
            
            # Debug first batch
            if batch_idx == 0:
                print(f"Debug - y_pred shape: {y_pred.shape}, min: {y_pred.min()}, max: {y_pred.max()}, unique: {np.unique(y_pred)}")
                print(f"Debug - y_true shape: {y_true_np.shape}, min: {y_true_np.min()}, max: {y_true_np.max()}, unique: {np.unique(y_true_np)}")
                print(f"Debug - y_true sum (non-zero pixels): {y_true_np.sum()}")
            
            Score = Eval_Score(y_pred, y_true_np)
            dice += Score.Dice()
            iou += Score.IoU()
            
            # Save images if save_dir is specified
            if save_dir and (max_images_to_save is None or batch_idx < max_images_to_save):
                # Get input image (for temporal mode, use the last frame)
                if args.use_temporal:
                    # x is [B, T, 3, H, W], get last frame
                    if x.dim() == 5:
                        input_img = x[0, -1].cpu()  # Last frame [3, H, W]
                    else:
                        input_img = x[0].cpu()
                else:
                    input_img = x[0].cpu()
                
                # Denormalize input image
                mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
                std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
                input_img = input_img * std + mean
                input_img = torch.clamp(input_img, 0, 1)
                input_img_np = (input_img.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                
                # Binary prediction (already 0 or 1)
                binary_pred_np = (y_pred * 255).astype(np.uint8)
                
                # Ground truth (use the binarized version)
                y_true_np_save = (y_true_np * 255).astype(np.uint8)
                
                # Instance prediction
                instance_pred = torch.squeeze(y['instance_seg_logits'].to('cpu')).numpy()
                if instance_pred.ndim == 3:
                    instance_pred_np = (instance_pred.transpose(1, 2, 0) * 255).astype(np.uint8)
                else:
                    instance_pred_np = (instance_pred * 255).astype(np.uint8)
                
                # Save images
                cv2.imwrite(os.path.join(save_dir, f'input_{batch_idx:04d}.jpg'), input_img_np)
                cv2.imwrite(os.path.join(save_dir, f'pred_binary_{batch_idx:04d}.jpg'), binary_pred_np)
                cv2.imwrite(os.path.join(save_dir, f'gt_binary_{batch_idx:04d}.jpg'), y_true_np_save)
                cv2.imwrite(os.path.join(save_dir, f'pred_instance_{batch_idx:04d}.jpg'), instance_pred_np)
                
                if batch_idx == 0:
                    print(f"Saving sample images to {save_dir}...")
    
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