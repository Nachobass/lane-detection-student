# coding: utf-8
"""
Coordinate Convolution (CoordConv) implementation
Adds explicit coordinate channels (X, Y) to input tensors
to help the model understand spatial relationships and geometry.

Reference: "An intriguing failing of convolutional neural networks and the CoordConv solution"
https://arxiv.org/abs/1807.03247
"""
import torch
import torch.nn as nn


class AddCoordinateChannels(nn.Module):
    """
    Adds coordinate channels (X, Y) to input tensors.
    Optionally adds radial distance channel (R) if with_r=True.
    
    The coordinate channels are normalized to [-1, 1] range.
    """
    def __init__(self, with_r=False):
        super().__init__()
        self.with_r = with_r

    def forward(self, input_tensor):
        """
        Args:
            input_tensor: shape (batch, channel, height, width)
        Returns:
            tensor: shape (batch, channel + 2, height, width) or 
                   (batch, channel + 3, height, width) if with_r=True
        """
        batch_size, _, height, width = input_tensor.size()

        # Create coordinate grids
        # X channel: varies along width dimension (0 to width-1)
        xx_channel = torch.arange(width, device=input_tensor.device, dtype=torch.float32).repeat(1, height, 1)
        # Y channel: varies along height dimension (0 to height-1)
        yy_channel = torch.arange(height, device=input_tensor.device, dtype=torch.float32).repeat(1, width, 1).transpose(1, 2)

        # Normalize to [-1, 1]
        if width > 1:
            xx_channel = (xx_channel / (width - 1)) * 2 - 1
        else:
            xx_channel = xx_channel * 0  # Single pixel case
        
        if height > 1:
            yy_channel = (yy_channel / (height - 1)) * 2 - 1
        else:
            yy_channel = yy_channel * 0  # Single pixel case

        # Expand to batch size
        xx_channel = xx_channel.expand(batch_size, 1, height, width)
        yy_channel = yy_channel.expand(batch_size, 1, height, width)

        # Ensure same device as input
        xx_channel = xx_channel.to(input_tensor.device)
        yy_channel = yy_channel.to(input_tensor.device)

        # Concatenate coordinate channels
        out = torch.cat([input_tensor, xx_channel, yy_channel], dim=1)

        # Optionally add radial distance channel
        if self.with_r:
            rr = torch.sqrt(torch.pow(xx_channel, 2) + torch.pow(yy_channel, 2))
            out = torch.cat([out, rr], dim=1)

        return out

