# coding: utf-8
"""
Training function for LaneNetPlus with multi-task support
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import time
import copy
import os
import cv2

from model.lanenet.loss import DiscriminativeLoss, FocalLoss, MultiTaskLoss


def compute_loss_plus(net_output, binary_label, instance_label, 
                      drivable_label=None, loss_type='FocalLoss', 
                      use_multitask=False, lambda_drivable=0.5):
    """
    Compute loss for LaneNetPlus with optional multi-task support
    
    Args:
        net_output: Model output dictionary
        binary_label: Ground truth binary lane mask
        instance_label: Ground truth instance mask
        drivable_label: Ground truth drivable area mask (optional)
        loss_type: Type of loss for lane segmentation
        use_multitask: Whether to use multi-task loss
        lambda_drivable: Weight for drivable area loss
    
    Returns:
        Tuple of (total_loss, lane_loss, instance_loss, drivable_loss, out)
    """
    k_binary = 10
    k_instance = 0.3
    k_dist = 1.0

    # Lane segmentation loss
    if loss_type == 'FocalLoss':
        loss_fn = FocalLoss(gamma=2, alpha=[0.25, 0.75])
        binary_seg_logits = net_output.get("binary_seg_logits", None)
        if binary_seg_logits is None:
            # Fallback to lane_logits if binary_seg_logits not available
            lane_logits = net_output["lane_logits"]
            # Convert to 2-channel for FocalLoss
            binary_seg_logits = torch.cat([1 - lane_logits, lane_logits], dim=1)
        binary_loss = loss_fn(binary_seg_logits, binary_label)
    elif loss_type == 'CrossEntropyLoss':
        loss_fn = nn.CrossEntropyLoss()
        binary_seg_logits = net_output.get("binary_seg_logits", None)
        if binary_seg_logits is None:
            lane_logits = net_output["lane_logits"]
            binary_seg_logits = torch.cat([1 - lane_logits, lane_logits], dim=1)
        binary_loss = loss_fn(binary_seg_logits, binary_label)
    else:
        # Use BCEWithLogitsLoss for binary segmentation
        loss_fn = nn.BCEWithLogitsLoss()
        lane_logits = net_output["lane_logits"]
        # Ensure binary_label is correct shape
        if len(binary_label.shape) == 3:
            binary_label_expanded = binary_label.unsqueeze(1).float()
        else:
            binary_label_expanded = binary_label.float()
        binary_loss = loss_fn(lane_logits, binary_label_expanded)

    # Instance segmentation loss
    pix_embedding = net_output["instance_seg_logits"]
    ds_loss_fn = DiscriminativeLoss(0.5, 1.5, 1.0, 1.0, 0.001)
    var_loss, dist_loss, reg_loss = ds_loss_fn(pix_embedding, instance_label)
    
    binary_loss = binary_loss * k_binary
    var_loss = var_loss * k_instance
    dist_loss = dist_loss * k_dist
    instance_loss = var_loss + dist_loss
    
    # Drivable area loss (if multi-task)
    drivable_loss = None
    if use_multitask and drivable_label is not None:
        if 'drivable_logits' in net_output:
            multitask_loss_fn = MultiTaskLoss(lambda_drivable=lambda_drivable)
            total_mt_loss, lane_mt_loss, drivable_mt_loss = multitask_loss_fn(
                net_output, binary_label, drivable_label
            )
            # Use multi-task loss components
            binary_loss = lane_mt_loss * k_binary
            drivable_loss = drivable_mt_loss
        else:
            # Fallback: use BCEWithLogitsLoss directly
            drivable_loss_fn = nn.BCEWithLogitsLoss()
            drivable_logits = net_output.get("drivable_logits")
            if drivable_logits is not None:
                if len(drivable_label.shape) == 3:
                    drivable_label_expanded = drivable_label.unsqueeze(1).float()
                else:
                    drivable_label_expanded = drivable_label.float()
                drivable_loss = drivable_loss_fn(drivable_logits, drivable_label_expanded)
    
    # Total loss
    if drivable_loss is not None:
        total_loss = binary_loss + instance_loss + lambda_drivable * drivable_loss
    else:
        total_loss = binary_loss + instance_loss
        drivable_loss = torch.tensor(0.0, device=binary_loss.device)
    
    out = net_output.get("binary_seg_pred", net_output.get("lane_pred"))
    
    return total_loss, binary_loss, instance_loss, drivable_loss, out


def train_model_plus(model, optimizer, scheduler, dataloaders, dataset_sizes, 
                     device, loss_type='FocalLoss', num_epochs=25,
                     use_multitask=False, lambda_drivable=0.5,
                     save_dir='./log', save_visualizations=False):
    """
    Train LaneNetPlus model with multi-task support
    
    Args:
        model: LaneNetPlus model
        optimizer: Optimizer
        scheduler: Learning rate scheduler
        dataloaders: Dictionary with 'train' and 'val' dataloaders
        dataset_sizes: Dictionary with dataset sizes
        device: Device to train on
        loss_type: Loss type for lane segmentation
        num_epochs: Number of training epochs
        use_multitask: Whether to use multi-task learning
        lambda_drivable: Weight for drivable area loss
        save_dir: Directory to save checkpoints and visualizations
        save_visualizations: Whether to save visualization images
    
    Returns:
        Tuple of (trained_model, training_log)
    """
    since = time.time()
    training_log = {
        'epoch': [],
        'training_loss': [],
        'training_lane_loss': [],
        'training_instance_loss': [],
        'training_drivable_loss': [],
        'val_loss': [],
        'val_lane_loss': [],
        'val_instance_loss': [],
        'val_drivable_loss': []
    }
    best_loss = float("inf")

    best_model_wts = copy.deepcopy(model.state_dict())
    
    # Create visualization directory if needed
    if save_visualizations:
        vis_dir = os.path.join(save_dir, 'visualizations')
        os.makedirs(vis_dir, exist_ok=True)

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
            running_loss_d = 0.0

            # Iterate over data
            for batch_idx, batch_data in enumerate(dataloaders[phase]):
                # Handle different batch formats (with or without drivable)
                if len(batch_data) == 4 and use_multitask:
                    inputs, binarys, instances, drivables = batch_data
                    drivables = drivables.type(torch.FloatTensor).to(device)
                else:
                    inputs, binarys, instances = batch_data[:3]
                    drivables = None
                
                inputs = inputs.type(torch.FloatTensor).to(device)
                binarys = binarys.type(torch.LongTensor).to(device)
                instances = instances.type(torch.FloatTensor).to(device)

                # Zero gradients
                optimizer.zero_grad()

                # Forward pass
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = compute_loss_plus(
                        outputs, binarys, instances, drivables,
                        loss_type=loss_type,
                        use_multitask=use_multitask,
                        lambda_drivable=lambda_drivable
                    )

                    # Backward + optimize only if in training phase
                    if phase == 'train':
                        loss[0].backward()
                        optimizer.step()

                # Statistics
                running_loss += loss[0].item() * inputs.size(0)
                running_loss_b += loss[1].item() * inputs.size(0)
                running_loss_i += loss[2].item() * inputs.size(0)
                running_loss_d += loss[3].item() * inputs.size(0)
                
                # Save visualizations occasionally
                if save_visualizations and phase == 'val' and batch_idx == 0:
                    _save_visualizations(outputs, inputs, binarys, drivables, 
                                        vis_dir, epoch, use_multitask)

            if phase == 'train':
                if scheduler is not None:
                    scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            binary_loss = running_loss_b / dataset_sizes[phase]
            instance_loss = running_loss_i / dataset_sizes[phase]
            drivable_loss = running_loss_d / dataset_sizes[phase]
            
            print('{} Total Loss: {:.4f} Lane Loss: {:.4f} Instance Loss: {:.4f} Drivable Loss: {:.4f}'.format(
                phase, epoch_loss, binary_loss, instance_loss, drivable_loss))

            # Log metrics
            if phase == 'train':
                training_log['training_loss'].append(epoch_loss)
                training_log['training_lane_loss'].append(binary_loss)
                training_log['training_instance_loss'].append(instance_loss)
                training_log['training_drivable_loss'].append(drivable_loss)
            if phase == 'val':
                training_log['val_loss'].append(epoch_loss)
                training_log['val_lane_loss'].append(binary_loss)
                training_log['val_instance_loss'].append(instance_loss)
                training_log['val_drivable_loss'].append(drivable_loss)
                if epoch_loss < best_loss:
                    best_loss = epoch_loss
                    best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val_loss: {:4f}'.format(best_loss))
    
    # Convert lists to numpy arrays
    for key in training_log:
        if key != 'epoch':
            training_log[key] = np.array(training_log[key])

    # Load best model weights
    model.load_state_dict(best_model_wts)
    return model, training_log


def _save_visualizations(outputs, inputs, binarys, drivables, vis_dir, epoch, use_multitask):
    """Save visualization images during training"""
    try:
        # Get first sample from batch
        input_img = inputs[0].cpu().numpy()
        lane_pred = outputs['lane_pred'][0, 0].cpu().numpy()
        binary_gt = binarys[0].cpu().numpy()
        
        # Denormalize input image
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        input_img = input_img.transpose(1, 2, 0)
        input_img = input_img * std + mean
        input_img = np.clip(input_img, 0, 1)
        input_img = (input_img * 255).astype(np.uint8)
        
        # Create visualization
        vis_img = input_img.copy()
        lane_overlay = np.zeros_like(vis_img)
        lane_overlay[:, :, 1] = (lane_pred > 0.5).astype(np.uint8) * 255
        vis_img = cv2.addWeighted(vis_img, 0.7, lane_overlay, 0.3, 0)
        
        # Save
        vis_path = os.path.join(vis_dir, f'epoch_{epoch:03d}_lane.png')
        cv2.imwrite(vis_path, cv2.cvtColor(vis_img, cv2.COLOR_RGB2BGR))
        
        # Save drivable if available
        if use_multitask and drivables is not None and 'drivable_pred' in outputs:
            drivable_pred = outputs['drivable_pred'][0, 0].cpu().numpy()
            drivable_overlay = np.zeros_like(vis_img)
            drivable_overlay[:, :, 0] = (drivable_pred > 0.5).astype(np.uint8) * 255
            vis_img_drivable = cv2.addWeighted(input_img, 0.7, drivable_overlay, 0.3, 0)
            vis_path_drivable = os.path.join(vis_dir, f'epoch_{epoch:03d}_drivable.png')
            cv2.imwrite(vis_path_drivable, cv2.cvtColor(vis_img_drivable, cv2.COLOR_RGB2BGR))
    except Exception as e:
        print(f"Warning: Could not save visualization: {e}")

