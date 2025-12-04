import time
import os
import sys

import torch
from model.lanenet.train_lanenet import train_model, train_temporal_model
from dataloader.data_loaders import TusimpleSet
from dataloader.sequence_dataset import SequenceDataset
from dataloader.transformers import Rescale
from model.lanenet.LaneNet import LaneNet
from torch.utils.data import DataLoader
from torch.autograd import Variable

from torchvision import transforms

from model.utils.cli_helper import parse_args
from model.eval_function import Eval_Score

import numpy as np
import pandas as pd
import cv2

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def train():
    args = parse_args()
    save_path = args.save
    if not os.path.isdir(save_path):
        os.makedirs(save_path)

    train_dataset_file = os.path.join(args.dataset, 'train.txt')
    val_dataset_file = os.path.join(args.dataset, 'val.txt')

    resize_height = args.height
    resize_width = args.width

    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize((resize_height, resize_width)),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize((resize_height, resize_width)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    target_transforms = transforms.Compose([
        Rescale((resize_width, resize_height)),
    ])

    # Check if temporal training is enabled
    if args.use_temporal:
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
            lr_phase2=args.lr * 0.1  # Lower learning rate for phase 2
        )
        
        # Create DataFrame with temporal training log
        df = pd.DataFrame({
            'epoch': log['epoch'],
            'phase': log['phase'],
            'training_loss': log['training_loss'],
            'val_loss': log['val_loss'],
            'val_iou': log.get('val_iou', [0] * len(log['epoch']))
        })
    else:
        # Standard non-temporal training
        train_dataset = TusimpleSet(train_dataset_file, transform=data_transforms['train'], target_transform=target_transforms)
        train_loader = DataLoader(train_dataset, batch_size=args.bs, shuffle=True)

        val_dataset = TusimpleSet(val_dataset_file, transform=data_transforms['val'], target_transform=target_transforms)
        val_loader = DataLoader(val_dataset, batch_size=args.bs, shuffle=True)

        dataloaders = {
            'train' : train_loader,
            'val' : val_loader
        }
        dataset_sizes = {'train': len(train_loader.dataset), 'val' : len(val_loader.dataset)}

        model = LaneNet(arch=args.model_type, use_temporal=False)
        model.to(DEVICE)

        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        print(f"{args.epochs} epochs {len(train_dataset)} training samples\n")

        model, log = train_model(model, optimizer, scheduler=None, dataloaders=dataloaders, dataset_sizes=dataset_sizes, device=DEVICE, loss_type=args.loss_type, num_epochs=args.epochs)
        
        df = pd.DataFrame({'epoch':[],'training_loss':[],'val_loss':[]})
        df['epoch'] = log['epoch']
        df['training_loss'] = log['training_loss']
        df['val_loss'] = log['val_loss']

    train_log_save_filename = os.path.join(save_path, 'training_log.csv')
    if args.use_temporal:
        df.to_csv(train_log_save_filename, columns=['epoch','phase','training_loss','val_loss','val_iou'], header=True,index=False,encoding='utf-8')
    else:
        df.to_csv(train_log_save_filename, columns=['epoch','training_loss','val_loss'], header=True,index=False,encoding='utf-8')
    print("training log is saved: {}".format(train_log_save_filename))
    
    model_save_filename = os.path.join(save_path, 'best_model.pth')
    torch.save(model.state_dict(), model_save_filename)
    print("model is saved: {}".format(model_save_filename))

if __name__ == '__main__':
    train()
