# coding: utf-8
"""
Multi-Task LaneNet Model
Extends LaneNet to predict both lane masks and drivable area masks
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.lanenet.LaneNet import LaneNet, DEVICE
from model.lanenet.backbone.UNet import UNet_Decoder
from model.lanenet.backbone.ENet import ENet_Decoder
from model.lanenet.backbone.deeplabv3_plus.deeplabv3plus import Deeplabv3plus_Decoder


class MultiTaskLaneNet(nn.Module):
    """
    Multi-Task LaneNet that predicts both lane masks and drivable area masks
    
    Architecture:
    - Shared encoder (from LaneNet)
    - Two separate decoder heads:
        - lane_head: predicts binary lane mask
        - drivable_head: predicts binary drivable area mask
    """
    
    def __init__(self, in_ch=3, arch="ENet", freeze_encoder=False):
        """
        Args:
            in_ch: Number of input channels (default: 3 for RGB)
            arch: Backbone architecture ('ENet', 'UNet', 'DeepLabv3+')
            freeze_encoder: Whether to freeze encoder weights
        """
        super(MultiTaskLaneNet, self).__init__()
        self._arch = arch
        self.freeze_encoder = freeze_encoder
        
        print(f"Initializing MultiTaskLaneNet with {arch} backbone")
        
        # Initialize encoder (shared between tasks)
        if self._arch == 'UNet':
            from model.lanenet.backbone.UNet import UNet_Encoder
            self._encoder = UNet_Encoder(in_ch)
            self._encoder.to(DEVICE)
            
            # Lane decoder: outputs 1 channel (binary mask)
            self._decoder_lane = UNet_Decoder(1)
            self._decoder_lane.to(DEVICE)
            
            # Drivable area decoder: outputs 1 channel (binary mask)
            self._decoder_drivable = UNet_Decoder(1)
            self._decoder_drivable.to(DEVICE)
            
        elif self._arch == 'ENet':
            from model.lanenet.backbone.ENet import ENet_Encoder
            self._encoder = ENet_Encoder(in_ch)
            self._encoder.to(DEVICE)
            
            # Lane decoder
            self._decoder_lane = ENet_Decoder(1)
            self._decoder_lane.to(DEVICE)
            
            # Drivable area decoder
            self._decoder_drivable = ENet_Decoder(1)
            self._decoder_drivable.to(DEVICE)
            
        elif self._arch == 'DeepLabv3+':
            from model.lanenet.backbone.deeplabv3_plus.deeplabv3plus import Deeplabv3plus_Encoder
            self._encoder = Deeplabv3plus_Encoder()
            self._encoder.to(DEVICE)
            
            # Lane decoder
            self._decoder_lane = Deeplabv3plus_Decoder(1)
            self._decoder_lane.to(DEVICE)
            
            # Drivable area decoder
            self._decoder_drivable = Deeplabv3plus_Decoder(1)
            self._decoder_drivable.to(DEVICE)
        else:
            raise ValueError(f"Unsupported architecture: {arch}")
        
        # Freeze encoder if requested
        if self.freeze_encoder:
            self.freeze_backbone()
    
    def freeze_backbone(self):
        """Freeze the encoder backbone parameters"""
        if hasattr(self, '_encoder'):
            for param in self._encoder.parameters():
                param.requires_grad = False
            print("Encoder backbone frozen - parameters will not be updated during training")
    
    def unfreeze_backbone(self):
        """Unfreeze the encoder backbone parameters"""
        if hasattr(self, '_encoder'):
            for param in self._encoder.parameters():
                param.requires_grad = True
            print("Encoder backbone unfrozen - parameters can be updated during training")
    
    def forward(self, input_tensor):
        """
        Forward pass through the multi-task model
        
        Args:
            input_tensor: Input image tensor (B, C, H, W)
        
        Returns:
            Dictionary with:
                - 'lane_logits': Raw logits for lane mask (B, 1, H, W)
                - 'lane_pred': Binary prediction for lane mask (B, 1, H, W)
                - 'drivable_logits': Raw logits for drivable area mask (B, 1, H, W)
                - 'drivable_pred': Binary prediction for drivable area mask (B, 1, H, W)
        """
        # Encode input
        if self._arch == 'UNet':
            c1, c2, c3, c4, c5 = self._encoder(input_tensor)
            lane_logits = self._decoder_lane(c1, c2, c3, c4, c5)
            drivable_logits = self._decoder_drivable(c1, c2, c3, c4, c5)
        elif self._arch == 'ENet':
            c = self._encoder(input_tensor)
            lane_logits = self._decoder_lane(c)
            drivable_logits = self._decoder_drivable(c)
        elif self._arch == 'DeepLabv3+':
            c1, c2 = self._encoder(input_tensor)
            lane_logits = self._decoder_lane(c1, c2)
            drivable_logits = self._decoder_drivable(c1, c2)
        else:
            raise ValueError(f"Unsupported architecture: {self._arch}")
        
        # Apply sigmoid to get probabilities
        lane_pred = torch.sigmoid(lane_logits)
        drivable_pred = torch.sigmoid(drivable_logits)
        
        # Binary predictions (threshold at 0.5)
        lane_binary = (lane_pred > 0.5).float()
        drivable_binary = (drivable_pred > 0.5).float()
        
        return {
            'lane_logits': lane_logits,
            'lane_pred': lane_binary,
            'lane_prob': lane_pred,
            'drivable_logits': drivable_logits,
            'drivable_pred': drivable_binary,
            'drivable_prob': drivable_pred
        }

