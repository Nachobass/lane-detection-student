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

    # Try to find test.txt, fallback to val.txt if not found
    test_file = os.path.join(args.dataset, 'test.txt')
    val_file = os.path.join(args.dataset, 'val.txt')
    if os.path.exists(test_file):
        dataset_file = test_file
    elif os.path.exists(val_file):
        dataset_file = val_file
        print(f"Note: test.txt not found, using val.txt instead")
    else:
        raise FileNotFoundError(f"Neither test.txt nor val.txt found in {args.dataset}")
    
    # Check if temporal mode is enabled
    if args.use_temporal:
        # Load image and mask paths from test.txt for SequenceDataset
        # Format: image_path binary_mask_path instance_mask_path
        test_image_paths = []
        test_mask_paths = []
        with open(dataset_file, 'r') as f:
            for line_idx, line in enumerate(f):
                parts = line.strip().split()
                if len(parts) >= 2:
                    test_image_paths.append(parts[0])
                    test_mask_paths.append(parts[1])  # Using binary mask path
                    # Debug first few paths
                    if line_idx < 3:
                        print(f"Debug - Line {line_idx}: image={parts[0]}, mask={parts[1]}")
                        if os.path.exists(parts[1]):
                            print(f"  Mask file exists: {parts[1]}")
                        else:
                            print(f"  WARNING: Mask file NOT found: {parts[1]}")
        
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
    checkpoint = torch.load(model_path, map_location=DEVICE)
    
    # Handle different checkpoint formats
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        # Full checkpoint format with model_state_dict, optimizer_state_dict, etc.
        state_dict = checkpoint['model_state_dict']
        print(f"Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")
        if 'best_val_loss' in checkpoint:
            print(f"Best validation loss: {checkpoint['best_val_loss']:.4f}")
    else:
        # Direct state_dict format
        state_dict = checkpoint
    
    # Load with strict=False to allow missing keys (e.g., ConvLSTM if checkpoint was from non-temporal training)
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    if missing_keys:
        print(f"Warning: Missing keys in checkpoint (will use default initialization):")
        for key in missing_keys[:5]:  
            print(f"  - {key}")
        if len(missing_keys) > 5:
            print(f"  ... and {len(missing_keys) - 5} more")
    if unexpected_keys:
        print(f"Warning: Unexpected keys in checkpoint (will be ignored):")
        for key in unexpected_keys[:5]:  
            print(f"  - {key}")
        if len(unexpected_keys) > 5:
            print(f"  ... and {len(unexpected_keys) - 5} more")
    
    model.eval()
    model.to(DEVICE)
    
    print(f"Model loaded from: {model_path}")
    print(f"Model type: {args.model_type}, Temporal: {args.use_temporal}")
    
    
    save_dir = args.save
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        print(f"Output images will be saved to: {save_dir}")
    
    
    max_images_to_save = 10  

    iou, dice = 0, 0
    with torch.no_grad():
        for batch_idx, batch_data in enumerate(eval_dataloader):
            if args.use_temporal:
                # Temporal mode: batch_data is (images, masks)
                x, masks = batch_data
                
                # Debug mask loading
                if batch_idx == 0:
                    print(f"Debug - masks shape from dataset: {masks.shape}")
                    print(f"Debug - masks min: {masks.min()}, max: {masks.max()}, unique: {torch.unique(masks)}")
                    print(f"Debug - masks sum: {masks.sum()}")
                
                # Reshape if needed: [B, T*3, H, W] -> [B, T, 3, H, W]
                if x.dim() == 4 and x.shape[1] == args.sequence_length * 3:
                    x = reshape_sequence_input(x, args.sequence_length)
                x = x.to(DEVICE)
                target = masks.to(DEVICE)
                
                # Handle mask shape
                if target.dim() == 4:
                    target = target.squeeze(1)  # [B, H, W]
                
                # Debug after shape handling
                if batch_idx == 0:
                    print(f"Debug - target shape after squeeze: {target.shape}")
                    print(f"Debug - target min: {target.min()}, max: {target.max()}, sum: {target.sum()}")
                    print(f"Debug - x shape after reshape: {x.shape}")
                    print(f"Debug - x min: {x.min()}, max: {x.max()}, mean: {x.mean()}")
            else:
                # Non-temporal mode: batch_data is (images, binary_masks, instance_masks)
                x, target, _ = batch_data
                x = x.to(DEVICE)
                target = target.to(DEVICE)
            
            y = model(x)
            
            # Debug model output
            if batch_idx == 0:
                print(f"Debug - model output keys: {y.keys()}")
                if 'binary_seg_logits' in y:
                    logits = y['binary_seg_logits']
                    print(f"Debug - binary_seg_logits shape: {logits.shape}, min: {logits.min()}, max: {logits.max()}, mean: {logits.mean()}")
            
            binary_pred = y['binary_seg_pred'].to('cpu')  # [B, 1, H, W] or [B, H, W]
            
            # Handle different shapes - remove all batch and channel dimensions
            # binary_pred is typically [B, 1, H, W] from argmax with keepdim=True
            if binary_pred.dim() == 4:
                # [B, 1, H, W] -> [B, H, W] -> [H, W] (take first item from batch)
                y_pred = binary_pred.squeeze(1)[0].numpy()  # Remove channel dim, then batch dim
            elif binary_pred.dim() == 3:
                # [B, H, W] -> [H, W] (take first item from batch)
                y_pred = binary_pred[0].numpy()
            elif binary_pred.dim() == 2:
                y_pred = binary_pred.numpy()
            else:
                y_pred = binary_pred.squeeze().numpy()
            
            y_true = target.to('cpu')
            # Remove batch dimension if present
            if y_true.dim() == 4:
                y_true = y_true.squeeze(1)[0]  # [B, 1, H, W] -> [B, H, W] -> [H, W]
            elif y_true.dim() == 3:
                y_true = y_true[0]  # [B, H, W] -> [H, W] (take first item from batch)
            elif y_true.dim() == 2:
                pass  # Already [H, W]
            else:
                y_true = y_true.squeeze()
            
            y_true_np = y_true.numpy() if isinstance(y_true, torch.Tensor) else y_true
            
            # Ensure both are 2D
            if y_pred.ndim > 2:
                y_pred = y_pred.squeeze()
            if y_true_np.ndim > 2:
                y_true_np = y_true_np.squeeze()
            
            if y_pred.shape != y_true_np.shape:
                print(f"WARNING: Shape mismatch! y_pred: {y_pred.shape}, y_true: {y_true_np.shape}")
                if len(y_pred.shape) == 2 and len(y_true_np.shape) == 2:
                    min_h = min(y_pred.shape[0], y_true_np.shape[0])
                    min_w = min(y_pred.shape[1], y_true_np.shape[1])
                    y_pred = y_pred[:min_h, :min_w]
                    y_true_np = y_true_np[:min_h, :min_w]
            
            # Binarize: values > 0.5 become 1, else 0
            # This handles both normalized masks [0, 1] and already binary masks
            y_true_np = (y_true_np > 0.5).astype(np.float32)
            
            # y_pred should already be 0 or 1 from argmax, but ensure it's float32
            y_pred = y_pred.astype(np.float32)
            
            # Debug first batch
            if batch_idx == 0:
                print(f"Debug - binary_pred tensor shape: {binary_pred.shape}")
                print(f"Debug - y_pred shape: {y_pred.shape}, min: {y_pred.min()}, max: {y_pred.max()}, unique: {np.unique(y_pred)}")
                print(f"Debug - y_true shape: {y_true_np.shape}, min: {y_true_np.min()}, max: {y_true_np.max()}, unique: {np.unique(y_true_np)}")
                print(f"Debug - y_true sum (non-zero pixels): {y_true_np.sum()}")
                print(f"Debug - y_pred sum (non-zero pixels): {(y_pred > 0.5).sum()}")
            
            # Ensure y_pred is in [0, 1] range (it should already be from argmax, but just in case)
            if y_pred.max() > 1.0 or y_pred.min() < 0.0:
                y_pred = np.clip(y_pred, 0, 1)
            
            Score = Eval_Score(y_pred, y_true_np)
            batch_iou = Score.IoU()
            batch_dice = Score.Dice()
            
            # Debug IoU calculation for first batch
            if batch_idx == 0:
                print(f"Debug - Batch IoU: {batch_iou:.4f}, Dice: {batch_dice:.4f}")
                print(f"Debug - Intersection: {Score.intersection}, Union: {Score.union}")
            
            dice += batch_dice
            iou += batch_iou
            
            if save_dir and (max_images_to_save is None or batch_idx < max_images_to_save):
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
                
                # Binary prediction (already 0 or 1) - ensure 2D for grayscale image
                binary_pred_np = (y_pred * 255).astype(np.uint8)
                if binary_pred_np.ndim > 2:
                    binary_pred_np = binary_pred_np.squeeze()
                # Ensure it's 2D (H, W) for grayscale
                assert binary_pred_np.ndim == 2, f"binary_pred_np should be 2D, got shape {binary_pred_np.shape}"
                
                # Ground truth (use the binarized version) - ensure 2D
                y_true_np_save = (y_true_np * 255).astype(np.uint8)
                if y_true_np_save.ndim > 2:
                    y_true_np_save = y_true_np_save.squeeze()
                assert y_true_np_save.ndim == 2, f"y_true_np_save should be 2D, got shape {y_true_np_save.shape}"
                
                # Instance prediction - handle different shapes
                instance_pred = y['instance_seg_logits'].to('cpu')
                # instance_pred is typically [B, C, H, W] where C is embedding dimension
                if instance_pred.dim() == 4:
                    instance_pred = instance_pred[0]  # [B, C, H, W] -> [C, H, W]
                if instance_pred.dim() == 3:
                    # [C, H, W] -> [H, W, C] for visualization
                    instance_pred_np = (instance_pred.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                elif instance_pred.dim() == 2:
                    # [H, W] -> [H, W, 1] for visualization
                    instance_pred_np = (instance_pred.numpy() * 255).astype(np.uint8)
                    instance_pred_np = np.expand_dims(instance_pred_np, axis=2)
                else:
                    instance_pred_np = (instance_pred.squeeze().numpy() * 255).astype(np.uint8)
                    if instance_pred_np.ndim == 2:
                        instance_pred_np = np.expand_dims(instance_pred_np, axis=2)
                
                # Ensure instance_pred_np is 3D (H, W, C) for RGB visualization
                if instance_pred_np.ndim == 2:
                    instance_pred_np = np.expand_dims(instance_pred_np, axis=2)
                # If it has more than 3 channels, take first 3
                if instance_pred_np.shape[2] > 3:
                    instance_pred_np = instance_pred_np[:, :, :3]
                # If it has 1 channel, convert to 3 channels (grayscale to RGB)
                elif instance_pred_np.shape[2] == 1:
                    instance_pred_np = np.repeat(instance_pred_np, 3, axis=2)
                
                cv2.imwrite(os.path.join(save_dir, f'input_{batch_idx:04d}.jpg'), input_img_np)
                cv2.imwrite(os.path.join(save_dir, f'pred_binary_{batch_idx:04d}.jpg'), binary_pred_np)
                cv2.imwrite(os.path.join(save_dir, f'gt_binary_{batch_idx:04d}.jpg'), y_true_np_save)
                cv2.imwrite(os.path.join(save_dir, f'pred_instance_{batch_idx:04d}.jpg'), instance_pred_np)
                
                if batch_idx == 0:
                    print(f"Saving sample images to {save_dir}...")
    
    final_iou = iou / len(eval_dataloader.dataset) if len(eval_dataloader.dataset) > 0 else 0.0
    final_dice = dice / len(eval_dataloader.dataset) if len(eval_dataloader.dataset) > 0 else 0.0
    


# ===== TuSimple Accuracy =====
    all_C = 0  # puntos correctos acumulados
    all_S = 0  # total de puntos GT acumulados

    print("Computing TuSimple lane accuracy...")

    with torch.no_grad():
        for batch_idx, batch_data in enumerate(eval_dataloader):
            if args.use_temporal:
                x, masks = batch_data
                if x.dim() == 4 and x.shape[1] == args.sequence_length * 3:
                    x = reshape_sequence_input(x, args.sequence_length)
                x = x.to(DEVICE)
                target = masks.to(DEVICE)
                if target.dim() == 4:
                    target = target.squeeze(1)
            else:
                x, target, _ = batch_data
                x = x.to(DEVICE)
                target = target.to(DEVICE)

            y = model(x)
            binary_pred = y['binary_seg_pred'].to('cpu')
            
            # Procesar predicción (binarizar)
            if binary_pred.dim() == 4:
                pred_mask = binary_pred.squeeze(1)[0].numpy()
            elif binary_pred.dim() == 3:
                pred_mask = binary_pred[0].numpy()
            else:
                pred_mask = binary_pred.numpy() if binary_pred.ndim == 2 else binary_pred.squeeze().numpy()
            
            # Procesar Ground Truth (binarizar)
            gt_mask = target.to('cpu')
            if gt_mask.dim() == 4:
                gt_mask = gt_mask.squeeze(1)[0].numpy()
            elif gt_mask.dim() == 3:
                gt_mask = gt_mask[0].numpy()
            else:
                gt_mask = gt_mask.numpy() if gt_mask.ndim == 2 else gt_mask.squeeze().numpy()
            
            # Asegurar dimensiones y tipos
            if pred_mask.ndim > 2: pred_mask = pred_mask.squeeze()
            if gt_mask.ndim > 2: gt_mask = gt_mask.squeeze()
            
            # Recortar si hay mismatch de tamaños
            min_h = min(pred_mask.shape[0], gt_mask.shape[0])
            min_w = min(pred_mask.shape[1], gt_mask.shape[1])
            pred_mask = pred_mask[:min_h, :min_w]
            gt_mask = gt_mask[:min_h, :min_w]

            gt_mask = (gt_mask > 0.5).astype(np.uint8)
            pred_mask = (pred_mask > 0.5).astype(np.uint8)
            
            H, W = gt_mask.shape
            
            # Obtenemos todas las coordenadas (y, x) donde hay carril real (GT)
            gt_ys, gt_xs = np.where(gt_mask > 0)
            
            # Total de puntos a evaluar en esta imagen
            S_i = len(gt_ys)
            C_i = 0
            
            for i in range(S_i):
                gy = gt_ys[i]
                gx = gt_xs[i]
                
                # Definimos ventana de tolerancia de 20 píxeles
                x_start = max(0, gx - 20)
                x_end = min(W, gx + 20)
                
                # Verificamos si la predicción tiene ALGÚN pixel blanco en esa ventana
                # en la misma fila 'gy'
                if np.any(pred_mask[gy, x_start:x_end] > 0):
                    C_i += 1
            
            all_S += S_i
            all_C += C_i

    tusimple_accuracy = all_C / all_S if all_S > 0 else 0.0

    print('='*60)
    print('EVALUATION RESULTS:')
    print('='*60)
    print(f'Pixel_IoU: {final_iou:.4f} ({final_iou*100:.2f}%)')
    print(f'Pixel_F1 (Dice): {final_dice:.4f} ({final_dice*100:.2f}%)')
    print(f'TuSimple_Accuracy: {tusimple_accuracy:.4f} ({tusimple_accuracy*100:.2f}%)')
    print('='*60)

if __name__ == "__main__":
    evaluation()