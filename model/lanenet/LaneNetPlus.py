# coding: utf-8
"""
LaneNetPlus: Enhanced LaneNet with Multi-Task Learning, Self-Attention, and Homography Support

This module integrates:
- Multi-task learning (lane + drivable area detection)
- Self-attention blocks in the encoder
- Support for homography-rectified images
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.lanenet.LaneNet import DEVICE
from model.lanenet.attention import SelfAttentionBlock
from model.lanenet.backbone.UNet import UNet_Encoder, UNet_Decoder
from model.lanenet.backbone.ENet import ENet_Encoder, ENet_Decoder
from model.lanenet.backbone.deeplabv3_plus.deeplabv3plus import Deeplabv3plus_Encoder, Deeplabv3plus_Decoder


class LaneNetPlus(nn.Module):
    """
    Enhanced LaneNet with multiple improvements:
    
    1. Multi-task learning: Predicts both lane masks and drivable area masks
    2. Self-attention: Optional attention blocks in encoder
    3. Homography support: Can work with rectified images
    
    Args:
        in_ch: Number of input channels (default: 3)
        arch: Backbone architecture ('ENet', 'UNet', 'DeepLabv3+')
        use_attention: Whether to use self-attention blocks
        use_multitask: Whether to predict drivable area (if False, only lanes)
        freeze_encoder: Whether to freeze encoder weights
    """
    
    def __init__(self, 
                 in_ch=3, 
                 arch="ENet",
                 use_attention=False,
                 use_multitask=True,
                 freeze_encoder=False):
        super(LaneNetPlus, self).__init__()
        
        self._arch = arch
        self.use_attention = use_attention
        self.use_multitask = use_multitask
        self.freeze_encoder = freeze_encoder
        
        print(f"Initializing LaneNetPlus:")
        print(f"  Architecture: {arch}")
        print(f"  Use Attention: {use_attention}")
        print(f"  Use Multi-Task: {use_multitask}")
        
        # Initialize encoder
        if self._arch == 'UNet':
            self._encoder = UNet_Encoder(in_ch)
            self._encoder.to(DEVICE)
            encoder_out_channels = 1024  # Last encoder output channels
            
        elif self._arch == 'ENet':
            self._encoder = ENet_Encoder(in_ch)
            self._encoder.to(DEVICE)
            encoder_out_channels = 128  # ENet encoder output
            
        elif self._arch == 'DeepLabv3+':
            self._encoder = Deeplabv3plus_Encoder()
            self._encoder.to(DEVICE)
            encoder_out_channels = 256  # Approximate
        else:
            raise ValueError(f"Unsupported architecture: {arch}")
        
        # Add self-attention blocks if requested
        if self.use_attention:
            if self._arch == 'ENet':
                # Insert attention after encoder output
                self.attention_block = SelfAttentionBlock(
                    embed_dim=encoder_out_channels,
                    num_heads=4,
                    dropout=0.1
                )
                self.attention_block.to(DEVICE)
            elif self._arch == 'UNet':
                # Add attention to the deepest feature map (c5)
                self.attention_block = SelfAttentionBlock(
                    embed_dim=encoder_out_channels,
                    num_heads=4,
                    dropout=0.1
                )
                self.attention_block.to(DEVICE)
            elif self._arch == 'DeepLabv3+':
                # Add attention to the main feature map
                self.attention_block = SelfAttentionBlock(
                    embed_dim=encoder_out_channels,
                    num_heads=4,
                    dropout=0.1
                )
                self.attention_block.to(DEVICE)
        
        # Lane decoder (always present)
        if self._arch == 'UNet':
            self._decoder_lane = UNet_Decoder(1)  # 1 channel for binary mask
            self._decoder_lane.to(DEVICE)
        elif self._arch == 'ENet':
            self._decoder_lane = ENet_Decoder(1)
            self._decoder_lane.to(DEVICE)
        elif self._arch == 'DeepLabv3+':
            self._decoder_lane = Deeplabv3plus_Decoder(1)
            self._decoder_lane.to(DEVICE)
        
        # Drivable area decoder (only if multi-task)
        if self.use_multitask:
            if self._arch == 'UNet':
                self._decoder_drivable = UNet_Decoder(1)
                self._decoder_drivable.to(DEVICE)
            elif self._arch == 'ENet':
                self._decoder_drivable = ENet_Decoder(1)
                self._decoder_drivable.to(DEVICE)
            elif self._arch == 'DeepLabv3+':
                self._decoder_drivable = Deeplabv3plus_Decoder(1)
                self._decoder_drivable.to(DEVICE)
        
        # Instance decoder (for backward compatibility)
        self.no_of_instances = 3
        if self._arch == 'UNet':
            self._decoder_instance = UNet_Decoder(self.no_of_instances)
            self._decoder_instance.to(DEVICE)
        elif self._arch == 'ENet':
            self._decoder_instance = ENet_Decoder(self.no_of_instances)
            self._decoder_instance.to(DEVICE)
        elif self._arch == 'DeepLabv3+':
            self._decoder_instance = Deeplabv3plus_Decoder(self.no_of_instances)
            self._decoder_instance.to(DEVICE)
        
        # Freeze encoder if requested
        if self.freeze_encoder:
            self.freeze_backbone()
    
    def freeze_backbone(self):
        """Freeze the encoder backbone parameters"""
        if hasattr(self, '_encoder'):
            for param in self._encoder.parameters():
                param.requires_grad = False
            print("Encoder backbone frozen")
    
    def unfreeze_backbone(self):
        """Unfreeze the encoder backbone parameters"""
        if hasattr(self, '_encoder'):
            for param in self._encoder.parameters():
                param.requires_grad = True
            print("Encoder backbone unfrozen")
    
    def forward(self, input_tensor):
        """
        Forward pass through LaneNetPlus
        
        Args:
            input_tensor: Input image tensor (B, C, H, W)
        
        Returns:
            Dictionary with predictions:
                - 'lane_logits': Raw logits for lane mask
                - 'lane_pred': Binary lane prediction
                - 'lane_prob': Lane probability map
                - 'drivable_logits': Raw logits for drivable area (if multitask)
                - 'drivable_pred': Binary drivable prediction (if multitask)
                - 'drivable_prob': Drivable probability map (if multitask)
                - 'instance_seg_logits': Instance segmentation (for compatibility)
                - 'binary_seg_logits': Binary segmentation logits (for compatibility)
                - 'binary_seg_pred': Binary segmentation prediction (for compatibility)
        """
        # Encode input
        if self._arch == 'UNet':
            c1, c2, c3, c4, c5 = self._encoder(input_tensor)
            
            # Apply attention to deepest features if enabled
            if self.use_attention:
                c5 = self.attention_block(c5)
            
            # Lane decoder
            lane_logits = self._decoder_lane(c1, c2, c3, c4, c5)
            
            # Drivable decoder (if multitask)
            if self.use_multitask:
                drivable_logits = self._decoder_drivable(c1, c2, c3, c4, c5)
            
            # Instance decoder (for compatibility)
            instance_logits = self._decoder_instance(c1, c2, c3, c4, c5)
            
        elif self._arch == 'ENet':
            c = self._encoder(input_tensor)
            
            # Apply attention if enabled
            if self.use_attention:
                c = self.attention_block(c)
            
            # Lane decoder
            lane_logits = self._decoder_lane(c)
            
            # Drivable decoder (if multitask)
            if self.use_multitask:
                drivable_logits = self._decoder_drivable(c)
            
            # Instance decoder (for compatibility)
            instance_logits = self._decoder_instance(c)
            
        elif self._arch == 'DeepLabv3+':
            c1, c2 = self._encoder(input_tensor)
            
            # Apply attention if enabled
            if self.use_attention:
                c1 = self.attention_block(c1)
            
            # Lane decoder
            lane_logits = self._decoder_lane(c1, c2)
            
            # Drivable decoder (if multitask)
            if self.use_multitask:
                drivable_logits = self._decoder_drivable(c1, c2)
            
            # Instance decoder (for compatibility)
            instance_logits = self._decoder_instance(c1, c2)
        else:
            raise ValueError(f"Unsupported architecture: {self._arch}")
        
        # Apply sigmoid to get probabilities
        lane_prob = torch.sigmoid(lane_logits)
        lane_pred = (lane_prob > 0.5).float()
        
        # Build output dictionary
        output = {
            'lane_logits': lane_logits,
            'lane_pred': lane_pred,
            'lane_prob': lane_prob,
            'instance_seg_logits': torch.sigmoid(instance_logits),
            'binary_seg_logits': torch.cat([1 - lane_logits, lane_logits], dim=1),  # For compatibility
            'binary_seg_pred': lane_pred  # For compatibility
        }
        
        # Add drivable outputs if multitask
        if self.use_multitask:
            drivable_prob = torch.sigmoid(drivable_logits)
            drivable_pred = (drivable_prob > 0.5).float()
            output['drivable_logits'] = drivable_logits
            output['drivable_pred'] = drivable_pred
            output['drivable_prob'] = drivable_prob
        
        return output

