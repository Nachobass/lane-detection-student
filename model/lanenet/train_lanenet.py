# coding: utf-8
"""
Training function for LaneNet with temporal ConvLSTM support
Supports two-phase training: Phase 1 (frozen encoder) and Phase 2 (full fine-tuning)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import time
import copy
import os
import argparse
from torch.utils.data import DataLoader

from model.lanenet.loss import DiscriminativeLoss, FocalLoss
from model.lanenet.LaneNet import LaneNet
from dataloader.sequence_dataset import SequenceDataset


def compute_loss(net_output, binary_label, instance_label, loss_type='FocalLoss'):
    """
    Compute loss for LaneNet output
    
    Args:
        net_output: Model output dictionary
        binary_label: Ground truth binary mask
        instance_label: Ground truth instance mask
        loss_type: Type of loss function
    
    Returns:
        Tuple of (total_loss, binary_loss, instance_loss, out)
    """
    # Loss weights - adjust these to balance binary vs instance segmentation
    # k_binary: Weight for binary segmentation loss (detecting lanes vs background)
    #   - Higher values (10-20) force model to focus more on detecting lane pixels
    #   - Current: 10 (already quite high)
    #   - If model struggles with lane detection, try increasing to 15-20
    # k_binary = 20  # Try 15-20 if lanes are not being detected well
    k_binary = 15
    
    # k_instance: Weight for instance segmentation loss (separating different lanes)
    #   - Lower values (0.3-0.5) are typical
    k_instance = 0.3
    k_dist = 1.0

    if loss_type == 'FocalLoss':
        loss_fn = FocalLoss(gamma=2, alpha=[0.25, 0.75])
    elif loss_type == 'CrossEntropyLoss':
        loss_fn = nn.CrossEntropyLoss()
    else:
        loss_fn = nn.CrossEntropyLoss()
    
    binary_seg_logits = net_output["binary_seg_logits"]
    binary_loss = loss_fn(binary_seg_logits, binary_label)

    pix_embedding = net_output["instance_seg_logits"]
    ds_loss_fn = DiscriminativeLoss(0.5, 1.5, 1.0, 1.0, 0.001)
    var_loss, dist_loss, reg_loss = ds_loss_fn(pix_embedding, instance_label)
    binary_loss_raw = binary_loss  # Save raw value before weighting
    binary_loss = binary_loss * k_binary
    var_loss = var_loss * k_instance
    dist_loss = dist_loss * k_dist
    instance_loss = var_loss + dist_loss
    total_loss = binary_loss + instance_loss
    out = net_output["binary_seg_pred"]

    return total_loss, binary_loss, instance_loss, out, binary_loss_raw


def compute_iou(pred, target):
    """
    Compute binary IoU between predictions and targets
    
    Args:
        pred: Predictions (logits or probabilities)
        target: Ground truth masks
    
    Returns:
        IoU score
    """
    if pred.dim() > 2:
        # If pred is logits, convert to binary
        if pred.shape[1] == 2:
            pred_binary = (torch.argmax(pred, dim=1) > 0).float()
        else:
            pred_binary = (pred > 0.5).float()
    else:
        pred_binary = (pred > 0.5).float()
    
    if target.dim() > 2:
        target_binary = (target > 0.5).float()
    else:
        target_binary = (target > 0.5).float()
    
    # Flatten for batch computation
    pred_flat = pred_binary.view(pred_binary.size(0), -1)
    target_flat = target_binary.view(target_binary.size(0), -1)
    
    intersection = (pred_flat * target_flat).sum(dim=1)
    union = pred_flat.sum(dim=1) + target_flat.sum(dim=1) - intersection
    
    iou = (intersection / (union + 1e-6)).mean()
    return iou.item()


def reshape_sequence_input(stacked_tensor, sequence_length):
    """
    Reshape stacked tensor from SequenceDataset [B, T*3, H, W] to [B, T, 3, H, W]
    
    Args:
        stacked_tensor: Tensor of shape [B, T*3, H, W]
        sequence_length: Number of frames T
    
    Returns:
        Reshaped tensor of shape [B, T, 3, H, W]
    """
    B, C, H, W = stacked_tensor.shape
    # Reshape: [B, T*3, H, W] -> [B, T, 3, H, W]
    return stacked_tensor.view(B, sequence_length, 3, H, W)


def train_temporal_model(
    model,
    train_loader,
    val_loader,
    device,
    loss_type='FocalLoss',
    num_epochs_phase1=5,
    num_epochs_phase2=20,
    freeze_encoder=True,
    save_dir='./log',
    lr_phase1=1e-3,
    lr_phase2=1e-4,
    pretrained_path=None
):
    """
    Train LaneNet with temporal support in two phases
    
    Phase 1: Train only ConvLSTM (encoder frozen)
    Phase 2: Fine-tune entire network (encoder + decoder + ConvLSTM)
    
    Args:
        model: LaneNet model with use_temporal=True
        train_loader: Training data loader
        val_loader: Validation data loader
        device: Device to train on
        loss_type: Loss function type
        num_epochs_phase1: Number of epochs for phase 1
        num_epochs_phase2: Number of epochs for phase 2
        freeze_encoder: Whether to freeze encoder in phase 1
        save_dir: Directory to save checkpoints
        lr_phase1: Learning rate for phase 1
        lr_phase2: Learning rate for phase 2
        pretrained_path: Path to pretrained model checkpoint (optional)
    
    Returns:
        Tuple of (trained_model, training_log)
    """
    since = time.time()
    training_log = {
        'epoch': [],
        'phase': [],
        'training_loss': [],
        'val_loss': [],
        'val_iou': []
    }
    best_loss = float("inf")
    best_model_wts = copy.deepcopy(model.state_dict())
    
    os.makedirs(save_dir, exist_ok=True)
    
    # Load pretrained model if provided
    if pretrained_path and os.path.exists(pretrained_path):
        print(f"\nLoading pretrained model from: {pretrained_path}")
        try:
            checkpoint = torch.load(pretrained_path, map_location=device)
            model.load_state_dict(checkpoint)
            best_model_wts = copy.deepcopy(model.state_dict())
            print("Pretrained model loaded successfully!")
            
            # If loading from a checkpoint, you might want to skip phase 1
            # This is handled by setting num_epochs_phase1=0 if needed
        except Exception as e:
            print(f"Warning: Could not load pretrained model: {e}")
            print("Starting training from scratch...")
    elif pretrained_path:
        print(f"Warning: Pretrained model path not found: {pretrained_path}")
        print("Starting training from scratch...")
    
    # ============================================================
    # PHASE 1: Train only ConvLSTM (encoder frozen)
    # ============================================================
    checkpoint_path = None  # Initialize to avoid UnboundLocalError
    if freeze_encoder and num_epochs_phase1 > 0:
        print("\n" + "="*60)
        print("PHASE 1: Training ConvLSTM with frozen encoder")
        print("="*60)
        
        # Freeze encoder - NO se entrenará en esta fase
        if hasattr(model, '_encoder'):
            for param in model._encoder.parameters():
                param.requires_grad = False  # ← Esto hace que NO se entrene
            # IMPORTANT: Keep encoder in train mode so BatchNorm works correctly
            # Even though params are frozen, BatchNorm needs to update running stats
            model._encoder.train()  # Modo train solo para BatchNorm, NO para actualizar pesos
            print("Encoder frozen (params) but kept in train mode (for BatchNorm)")
        
        # Freeze decoder (optional - uncomment if needed)
        # for param in model._decoder_binary.parameters():
        #     param.requires_grad = False
        # for param in model._decoder_instance.parameters():
        #     param.requires_grad = False
        
        # Only train ConvLSTM (and decoder if not frozen)
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        optimizer = optim.Adam(trainable_params, lr=lr_phase1)
        print(f"Training {sum(p.numel() for p in trainable_params)} parameters")
        
        # Training loop Phase 1
        for epoch in range(num_epochs_phase1):
            training_log['epoch'].append(epoch)
            training_log['phase'].append(1)
            print(f'\nPhase 1 - Epoch {epoch}/{num_epochs_phase1 - 1}')
            print('-' * 10)
            
            # Training phase
            model.train()
            running_loss = 0.0
            running_loss_b = 0.0
            running_loss_i = 0.0
            running_binary_loss_raw = 0.0
            
            for batch_idx, (images, masks) in enumerate(train_loader):
                # Reshape if needed: [B, T*3, H, W] -> [B, T, 3, H, W]
                # The model can handle both formats, but we reshape here for consistency
                if images.dim() == 4 and images.shape[1] == model.sequence_length * 3:
                    images = reshape_sequence_input(images, model.sequence_length)
                
                images = images.to(device)
                masks = masks.to(device)
                
                # Debug: Verify input shape (only first batch of first epoch)
                if epoch == 0 and batch_idx == 0:
                    print(f"Debug - Input shape: {images.shape}, Expected: [B, {model.sequence_length}, 3, H, W]")
                    print(f"Debug - Mask shape: {masks.shape}")
                
                # Handle mask shape: [B, 1, H, W] or [B, H, W]
                if masks.dim() == 4:
                    binary_masks = masks.squeeze(1).long()  # [B, H, W]
                else:
                    binary_masks = masks.long()
                
                # Create instance mask (same as binary for now)
                instance_masks = masks.float()
                
                optimizer.zero_grad()
                
                # Forward pass
                outputs = model(images)
                loss = compute_loss(outputs, binary_masks, instance_masks, loss_type)
                total_loss, binary_loss, instance_loss, out, binary_loss_raw = loss
                
                # Backward pass
                total_loss.backward()
                
                # Debug: Check if gradients are flowing (only first batch of first epoch)
                # IMPORTANT: Check AFTER backward() but BEFORE optimizer.step()
                if epoch == 0 and batch_idx == 0:
                    # Check if ConvLSTM has gradients
                    if hasattr(model, 'conv_lstm'):
                        conv_lstm_grads = []
                        for name, param in model.conv_lstm.named_parameters():
                            if param.grad is not None:
                                conv_lstm_grads.append(param.grad.abs().mean().item())
                        if conv_lstm_grads:
                            avg_grad = sum(conv_lstm_grads) / len(conv_lstm_grads)
                            print(f"Debug - ConvLSTM gradient magnitude: {avg_grad:.6f} (checked {len(conv_lstm_grads)} params)")
                        else:
                            print("Debug - WARNING: No gradients in ConvLSTM!")
                            # Try to check if encoder outputs have gradients
                            print("Debug - Checking if encoder outputs have gradients...")
                            # Check decoder gradients as fallback
                            decoder_grads = []
                            for name, param in model._decoder_binary.named_parameters():
                                if param.grad is not None:
                                    decoder_grads.append(param.grad.abs().mean().item())
                            if decoder_grads:
                                print(f"Debug - Decoder has gradients: {sum(decoder_grads)/len(decoder_grads):.6f}")
                            else:
                                print("Debug - ERROR: No gradients anywhere!")
                
                # Gradient clipping to prevent exploding gradients
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                optimizer.step()
                
                # Statistics
                running_loss += total_loss.item() * images.size(0)
                running_loss_b += binary_loss.item() * images.size(0)
                running_loss_i += instance_loss.item() * images.size(0)
                # Also track raw binary loss for display
                if epoch == 0 and batch_idx == 0:
                    running_binary_loss_raw = binary_loss_raw.item() * images.size(0)
                else:
                    running_binary_loss_raw += binary_loss_raw.item() * images.size(0)
            
            epoch_loss = running_loss / len(train_loader.dataset)
            binary_loss = running_loss_b / len(train_loader.dataset)
            instance_loss = running_loss_i / len(train_loader.dataset)
            binary_loss_raw_avg = running_binary_loss_raw / len(train_loader.dataset)
            
            # Validation
            model.eval()
            val_loss = 0.0
            val_iou = 0.0
            
            with torch.no_grad():
                for images, masks in val_loader:
                    if images.dim() == 4 and images.shape[1] == model.sequence_length * 3:
                        images = reshape_sequence_input(images, model.sequence_length)
                    
                    images = images.to(device)
                    masks = masks.to(device)
                    
                    if masks.dim() == 4:
                        binary_masks = masks.squeeze(1).long()
                    else:
                        binary_masks = masks.long()
                    
                    instance_masks = masks.float()
                    
                    outputs = model(images)
                    loss = compute_loss(outputs, binary_masks, instance_masks, loss_type)
                    total_loss_val, _, _, _, _ = loss
                    val_loss += total_loss_val.item() * images.size(0)
                    
                    # Compute IoU
                    iou = compute_iou(outputs['binary_seg_logits'], binary_masks)
                    val_iou += iou * images.size(0)
            
            val_loss = val_loss / len(val_loader.dataset)
            val_iou = val_iou / len(val_loader.dataset)
            
            # Show weighted and unweighted losses for better understanding
            print(f'Train Loss: {epoch_loss:.4f} (Binary: {binary_loss:.4f} [raw: {binary_loss_raw_avg:.4f}], Instance: {instance_loss:.4f})')
            print(f'Val Loss: {val_loss:.4f}, Val IoU: {val_iou:.4f} ({val_iou*100:.2f}%)')
            
            training_log['training_loss'].append(epoch_loss)
            training_log['val_loss'].append(val_loss)
            training_log['val_iou'].append(val_iou)
            
            # Save checkpoint
            if val_loss < best_loss:
                best_loss = val_loss
                best_model_wts = copy.deepcopy(model.state_dict())
            
            # Save phase 1 checkpoint
            checkpoint_path = os.path.join(save_dir, 'ckpt_phase1.pth')
            torch.save(model.state_dict(), checkpoint_path)
        
        print(f"\nPhase 1 complete. Best val loss: {best_loss:.4f}")
        if checkpoint_path:
            print(f"Checkpoint saved: {checkpoint_path}")
    elif num_epochs_phase1 == 0:
        print("\n" + "="*60)
        print("PHASE 1: Skipped (num_epochs_phase1=0)")
        print("="*60)
        print("Proceeding directly to Phase 2...")
    
    # ============================================================
    # PHASE 2: Fine-tune entire network
    # ============================================================
    print("\n" + "="*60)
    print("PHASE 2: Fine-tuning entire network")
    print("="*60)
    
    # Unfreeze encoder and decoder - AHORA SÍ SE ENTRENAN
    if hasattr(model, '_encoder'):
        for param in model._encoder.parameters():
            param.requires_grad = True
        print("Encoder unfrozen")
    
    for param in model._decoder_binary.parameters():
        param.requires_grad = True
    for param in model._decoder_instance.parameters():
        param.requires_grad = True
    
    # Create new optimizer with all parameters
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.Adam(trainable_params, lr=lr_phase2)
    
    # Add learning rate scheduler for phase 2 (reduce LR when plateau)
    scheduler = lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3, min_lr=1e-6
    )
    
    print(f"Training {sum(p.numel() for p in trainable_params)} parameters")
    print(f"Using ReduceLROnPlateau scheduler (factor=0.5, patience=3)")
    
    # Load best model from phase 1
    model.load_state_dict(best_model_wts)
    best_loss = float("inf")
    best_iou = 0.0  # Track best IoU for early stopping
    
    # Training loop Phase 2
    for epoch in range(num_epochs_phase2):
        training_log['epoch'].append(num_epochs_phase1 + epoch)
        training_log['phase'].append(2)
        print(f'\nPhase 2 - Epoch {epoch}/{num_epochs_phase2 - 1}')
        print('-' * 10)
        
        # Training phase
        model.train()
        running_loss = 0.0
        running_loss_b = 0.0
        running_loss_i = 0.0
        running_binary_loss_raw = 0.0
        
        for batch_idx, (images, masks) in enumerate(train_loader):
            if images.dim() == 4 and images.shape[1] == model.sequence_length * 3:
                images = reshape_sequence_input(images, model.sequence_length)
            
            images = images.to(device)
            masks = masks.to(device)
            
            if masks.dim() == 4:
                binary_masks = masks.squeeze(1).long()
            else:
                binary_masks = masks.long()
            
            instance_masks = masks.float()
            
            optimizer.zero_grad()
            
            outputs = model(images)
            loss = compute_loss(outputs, binary_masks, instance_masks, loss_type)
            total_loss, binary_loss, instance_loss, out, binary_loss_raw = loss
            
            total_loss.backward()
            optimizer.step()
            
            running_loss += total_loss.item() * images.size(0)
            running_loss_b += binary_loss.item() * images.size(0)
            running_loss_i += instance_loss.item() * images.size(0)
            running_binary_loss_raw += binary_loss_raw.item() * images.size(0)
        
        epoch_loss = running_loss / len(train_loader.dataset)
        binary_loss = running_loss_b / len(train_loader.dataset)
        instance_loss = running_loss_i / len(train_loader.dataset)
        binary_loss_raw_avg = running_binary_loss_raw / len(train_loader.dataset)
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_iou = 0.0
        
        with torch.no_grad():
            for images, masks in val_loader:
                if images.dim() == 4 and images.shape[1] == model.sequence_length * 3:
                    images = reshape_sequence_input(images, model.sequence_length)
                
                images = images.to(device)
                masks = masks.to(device)
                
                if masks.dim() == 4:
                    binary_masks = masks.squeeze(1).long()
                else:
                    binary_masks = masks.long()
                
                instance_masks = masks.float()
                
                outputs = model(images)
                loss = compute_loss(outputs, binary_masks, instance_masks, loss_type)
                total_loss_val, _, _, _, _ = loss
                val_loss += total_loss_val.item() * images.size(0)
                
                iou = compute_iou(outputs['binary_seg_logits'], binary_masks)
                val_iou += iou * images.size(0)
        
        val_loss = val_loss / len(val_loader.dataset)
        val_iou = val_iou / len(val_loader.dataset)
        
        # Update learning rate scheduler
        old_lr = optimizer.param_groups[0]['lr']
        scheduler.step(val_loss)
        new_lr = optimizer.param_groups[0]['lr']
        if new_lr < old_lr:
            print(f'  -> Learning rate reduced: {old_lr:.6f} -> {new_lr:.6f}')
        
        # Show weighted and unweighted losses for better understanding
        binary_loss_raw_avg = running_binary_loss_raw / len(train_loader.dataset)
        current_lr = optimizer.param_groups[0]['lr']
        print(f'Train Loss: {epoch_loss:.4f} (Binary: {binary_loss:.4f} [raw: {binary_loss_raw_avg:.4f}], Instance: {instance_loss:.4f})')
        print(f'Val Loss: {val_loss:.4f}, Val IoU: {val_iou:.4f} ({val_iou*100:.2f}%), LR: {current_lr:.6f}')
        
        training_log['training_loss'].append(epoch_loss)
        training_log['val_loss'].append(val_loss)
        training_log['val_iou'].append(val_iou)
        
        # Save checkpoint based on IoU (better metric than loss for segmentation)
        if val_iou > best_iou:
            best_iou = val_iou
            best_loss = val_loss
            best_model_wts = copy.deepcopy(model.state_dict())
            print(f'  -> New best IoU: {best_iou:.4f} ({best_iou*100:.2f}%)')
        
        # Save periodic checkpoints
        if (epoch + 1) % 5 == 0 or epoch == num_epochs_phase2 - 1:
            checkpoint_path = os.path.join(save_dir, f'ckpt_phase2_epoch{epoch+1}.pth')
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Checkpoint saved: {checkpoint_path}")
    
    time_elapsed = time.time() - since
    print(f'\nTraining complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val_loss: {best_loss:.4f}')
    
    # Convert lists to numpy arrays
    for key in training_log:
        if key != 'phase':
            training_log[key] = np.array(training_log[key])
    
    # Load best model weights
    model.load_state_dict(best_model_wts)
    return model, training_log


def train_model(model, optimizer, scheduler, dataloaders, dataset_sizes, device, loss_type='FocalLoss', num_epochs=25):
    """
    Original training function for backward compatibility (non-temporal mode)
    """
    since = time.time()
    training_log = {'epoch': [], 'training_loss': [], 'val_loss': []}
    best_loss = float("inf")

    best_model_wts = copy.deepcopy(model.state_dict())

    for epoch in range(num_epochs):
        training_log['epoch'].append(epoch)
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_loss_b = 0.0
            running_loss_i = 0.0

            # Iterate over data.
            for inputs, binarys, instances in dataloaders[phase]:
                inputs = inputs.type(torch.FloatTensor).to(device)
                binarys = binarys.type(torch.LongTensor).to(device)
                instances = instances.type(torch.FloatTensor).to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = compute_loss(outputs, binarys, instances, loss_type)
                    total_loss, binary_loss, instance_loss, out, _ = loss

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        total_loss.backward()
                        optimizer.step()

                # statistics
                running_loss += total_loss.item() * inputs.size(0)
                running_loss_b += binary_loss.item() * inputs.size(0)
                running_loss_i += instance_loss.item() * inputs.size(0)

            if phase == 'train':
                if scheduler != None:
                    scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            binary_loss = running_loss_b / dataset_sizes[phase]
            instance_loss = running_loss_i / dataset_sizes[phase]
            print('{} Total Loss: {:.4f} Binary Loss: {:.4f} Instance Loss: {:.4f}'.format(
                phase, epoch_loss, binary_loss, instance_loss))

            # deep copy the model
            if phase == 'train':
                training_log['training_loss'].append(epoch_loss)
            if phase == 'val':
                training_log['val_loss'].append(epoch_loss)
                if epoch_loss < best_loss:
                    best_loss = epoch_loss
                    best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val_loss: {:4f}'.format(best_loss))
    training_log['training_loss'] = np.array(training_log['training_loss'])
    training_log['val_loss'] = np.array(training_log['val_loss'])

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, training_log


def trans_to_cuda(variable):
    """Helper function to move variable to CUDA"""
    if torch.cuda.is_available():
        return variable.cuda()
    else:
        return variable


if __name__ == '__main__':
    """
    Direct execution of train_lanenet.py for temporal training
    Usage: python -m model.lanenet.train_lanenet --dataset ./data/training_data_example --use_temporal
    """
    import sys
    import os
    from torch.utils.data import DataLoader
    
    # Add parent directories to path for imports
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../..'))
    
    from model.utils.cli_helper import parse_args
    from dataloader.sequence_dataset import SequenceDataset
    from dataloader.data_loaders import TusimpleSet
    from dataloader.transformers import Rescale
    from torchvision import transforms
    import pandas as pd
    
    DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    args = parse_args()
    
    if not args.use_temporal:
        print("ERROR: train_lanenet.py is designed for temporal training.")
        print("Please use --use_temporal flag or use train.py for standard training.")
        sys.exit(1)
    
    if not args.dataset:
        print("ERROR: --dataset argument is required")
        sys.exit(1)
    
    save_path = args.save
    if not os.path.isdir(save_path):
        os.makedirs(save_path)
    
    train_dataset_file = os.path.join(args.dataset, 'train.txt')
    val_dataset_file = os.path.join(args.dataset, 'val.txt')
    
    resize_height = args.height
    resize_width = args.width
    
    # Load image and mask paths from train.txt and val.txt for SequenceDataset
    train_image_paths = []
    train_mask_paths = []
    with open(train_dataset_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                train_image_paths.append(parts[0])
                train_mask_paths.append(parts[1])  # Using binary mask path
    
    val_image_paths = []
    val_mask_paths = []
    with open(val_dataset_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                val_image_paths.append(parts[0])
                val_mask_paths.append(parts[1])  # Using binary mask path
    
    # Create SequenceDataset for temporal training
    train_dataset = SequenceDataset(
        train_image_paths, 
        train_mask_paths, 
        sequence_len=args.sequence_length,
        target_size=(resize_width, resize_height)
    )
    train_loader = DataLoader(train_dataset, batch_size=args.bs, shuffle=True)
    
    val_dataset = SequenceDataset(
        val_image_paths,
        val_mask_paths,
        sequence_len=args.sequence_length,
        target_size=(resize_width, resize_height)
    )
    val_loader = DataLoader(val_dataset, batch_size=args.bs, shuffle=False)
    
    # Create model with temporal support
    model = LaneNet(arch=args.model_type, use_temporal=True, sequence_length=args.sequence_length)
    model.to(DEVICE)
    
    print(f"Temporal training enabled with sequence length: {args.sequence_length}")
    print(f"Phase 1: {args.num_epochs_phase1} epochs, Phase 2: {args.num_epochs_phase2} epochs")
    if args.pretrained:
        print(f"Will load pretrained model from: {args.pretrained}")
        if args.num_epochs_phase1 == 0:
            print("Note: Phase 1 skipped (num_epochs_phase1=0), will only fine-tune in Phase 2")
    print(f"{len(train_dataset)} training samples\n")
    
    # Use temporal training function
    model, log = train_temporal_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=DEVICE,
        loss_type=args.loss_type,
        num_epochs_phase1=args.num_epochs_phase1,
        num_epochs_phase2=args.num_epochs_phase2,
        freeze_encoder=args.freeze_encoder,
        save_dir=save_path,
        lr_phase1=args.lr,
        lr_phase2=args.lr * 0.1,  # Lower learning rate for phase 2
        pretrained_path=args.pretrained
    )
    
    # Create DataFrame with temporal training log
    df = pd.DataFrame({
        'epoch': log['epoch'],
        'phase': log['phase'],
        'training_loss': log['training_loss'],
        'val_loss': log['val_loss'],
        'val_iou': log.get('val_iou', [0] * len(log['epoch']))
    })
    
    train_log_save_filename = os.path.join(save_path, 'training_log.csv')
    df.to_csv(train_log_save_filename, columns=['epoch','phase','training_loss','val_loss','val_iou'], header=True,index=False,encoding='utf-8')
    print("training log is saved: {}".format(train_log_save_filename))
    
    model_save_filename = os.path.join(save_path, 'best_model.pth')
    torch.save(model.state_dict(), model_save_filename)
    print("model is saved: {}".format(model_save_filename))
