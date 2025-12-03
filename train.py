import time
import os
import sys

import torch
from model.lanenet.train_lanenet import train_model
from model.lanenet.train_lanenet_plus import train_model_plus
from dataloader.data_loaders import TusimpleSet
from dataloader.transformers import Rescale
from model.lanenet.LaneNet import LaneNet
from model.lanenet.LaneNetPlus import LaneNetPlus
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

    # Create datasets with optional drivable area support
    train_dataset = TusimpleSet(
        train_dataset_file, 
        transform=data_transforms['train'], 
        target_transform=target_transforms,
        drivable_dir=args.drivable_dir if args.use_multitask else None,
        use_drivable=args.use_multitask
    )
    train_loader = DataLoader(train_dataset, batch_size=args.bs, shuffle=True)

    val_dataset = TusimpleSet(
        val_dataset_file, 
        transform=data_transforms['val'], 
        target_transform=target_transforms,
        drivable_dir=args.drivable_dir if args.use_multitask else None,
        use_drivable=args.use_multitask
    )
    val_loader = DataLoader(val_dataset, batch_size=args.bs, shuffle=True)

    dataloaders = {
        'train' : train_loader,
        'val' : val_loader
    }
    dataset_sizes = {'train': len(train_loader.dataset), 'val' : len(val_loader.dataset)}

    # Create model (LaneNet or LaneNetPlus)
    if args.use_lanenet_plus:
        model = LaneNetPlus(
            arch=args.model_type,
            use_attention=args.use_attention,
            use_multitask=args.use_multitask,
            freeze_encoder=False
        )
    else:
        model = LaneNet(arch=args.model_type)
    
    model.to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    print(f"{args.epochs} epochs {len(train_dataset)} training samples\n")
    print(f"Model: {'LaneNetPlus' if args.use_lanenet_plus else 'LaneNet'}")
    if args.use_lanenet_plus:
        print(f"  - Attention: {args.use_attention}")
        print(f"  - Multi-task: {args.use_multitask}")
        print(f"  - Rectification: {args.use_rectification}")

    # Train model
    if args.use_lanenet_plus:
        model, log = train_model_plus(
            model, optimizer, scheduler=None, 
            dataloaders=dataloaders, dataset_sizes=dataset_sizes, 
            device=DEVICE, loss_type=args.loss_type, num_epochs=args.epochs,
            use_multitask=args.use_multitask, lambda_drivable=args.lambda_drivable,
            save_dir=save_path, save_visualizations=args.save_visualizations
        )
        
        # Save extended log
        df = pd.DataFrame({
            'epoch': log['epoch'],
            'training_loss': log['training_loss'],
            'training_lane_loss': log['training_lane_loss'],
            'training_instance_loss': log['training_instance_loss'],
            'training_drivable_loss': log['training_drivable_loss'],
            'val_loss': log['val_loss'],
            'val_lane_loss': log['val_lane_loss'],
            'val_instance_loss': log['val_instance_loss'],
            'val_drivable_loss': log['val_drivable_loss']
        })
    else:
        model, log = train_model(
            model, optimizer, scheduler=None, 
            dataloaders=dataloaders, dataset_sizes=dataset_sizes, 
            device=DEVICE, loss_type=args.loss_type, num_epochs=args.epochs
        )
        
        # Save standard log
        df = pd.DataFrame({
            'epoch': log['epoch'],
            'training_loss': log['training_loss'],
            'val_loss': log['val_loss']
        })

    train_log_save_filename = os.path.join(save_path, 'training_log.csv')
    df.to_csv(train_log_save_filename, index=False, encoding='utf-8')
    print("training log is saved: {}".format(train_log_save_filename))
    
    model_save_filename = os.path.join(save_path, 'best_model.pth')
    torch.save(model.state_dict(), model_save_filename)
    print("model is saved: {}".format(model_save_filename))

if __name__ == '__main__':
    train()
