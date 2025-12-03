# coding: utf-8
"""
Self-Attention Block for LaneNet Encoder
Implements Multi-Head Self-Attention adapted for 2D feature maps
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class SelfAttentionBlock(nn.Module):
    """
    Self-Attention Block for 2D feature maps
    
    Adapts Multi-Head Attention to work with spatial feature maps by:
    1. Flattening spatial dimensions
    2. Applying multi-head attention
    3. Reshaping back to spatial dimensions
    """
    
    def __init__(self, embed_dim, num_heads=4, dropout=0.1):
        """
        Args:
            embed_dim: Dimension of input features (channels)
            num_heads: Number of attention heads
            dropout: Dropout probability
        """
        super(SelfAttentionBlock, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        
        # Multi-head attention
        self.attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=False  # (seq_len, batch, embed_dim)
        )
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim),
            nn.Dropout(dropout)
        )
        
    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (B, C, H, W)
        
        Returns:
            Output tensor of shape (B, C, H, W)
        """
        B, C, H, W = x.shape
        
        # Save residual
        residual = x
        
        # Flatten spatial dimensions: (B, C, H, W) -> (B, C, H*W) -> (H*W, B, C)
        x_flat = x.view(B, C, H * W).permute(2, 0, 1)  # (H*W, B, C)
        
        # Apply layer norm
        x_norm = self.norm1(x_flat)
        
        # Self-attention: query, key, value are all the same
        attn_out, _ = self.attention(x_norm, x_norm, x_norm)
        
        # Residual connection
        x_attn = x_flat + attn_out
        
        # Apply layer norm
        x_attn_norm = self.norm2(x_attn)
        
        # Feed-forward network
        ffn_out = self.ffn(x_attn_norm)
        
        # Residual connection
        x_out = x_attn + ffn_out
        
        # Reshape back to spatial: (H*W, B, C) -> (B, C, H*W) -> (B, C, H, W)
        x_out = x_out.permute(1, 2, 0).view(B, C, H, W)
        
        # Final residual connection with original input
        output = residual + x_out
        
        return output


class SpatialSelfAttention(nn.Module):
    """
    Spatial Self-Attention that processes features in a more efficient way
    by applying attention along spatial dimensions separately
    """
    
    def __init__(self, in_channels, reduction=8):
        """
        Args:
            in_channels: Number of input channels
            reduction: Reduction factor for channel attention
        """
        super(SpatialSelfAttention, self).__init__()
        self.in_channels = in_channels
        
        # Channel attention
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, 1),
            nn.Sigmoid()
        )
        
        # Spatial attention
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (B, C, H, W)
        
        Returns:
            Output tensor of shape (B, C, H, W)
        """
        # Channel attention
        ca = self.channel_attention(x)
        x_ca = x * ca
        
        # Spatial attention
        avg_out = torch.mean(x_ca, dim=1, keepdim=True)
        max_out, _ = torch.max(x_ca, dim=1, keepdim=True)
        spatial_input = torch.cat([avg_out, max_out], dim=1)
        sa = self.spatial_attention(spatial_input)
        x_out = x_ca * sa
        
        return x_out

