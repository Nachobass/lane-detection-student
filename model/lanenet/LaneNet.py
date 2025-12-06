# coding: utf-8
"""
LaneNet model
https://arxiv.org/pdf/1807.01726.pdf
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.lanenet.loss import DiscriminativeLoss
from model.lanenet.backbone.UNet import UNet_Encoder, UNet_Decoder
from model.lanenet.backbone.ENet import ENet_Encoder, ENet_Decoder
from model.lanenet.backbone.deeplabv3_plus.deeplabv3plus import Deeplabv3plus_Encoder, Deeplabv3plus_Decoder
from model.lanenet.backbone.temporal_conv_lstm import ConvLSTM

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class LaneNet(nn.Module):
    def __init__(self, in_ch = 3, arch="ENet", use_temporal=False, sequence_length=3):
        super(LaneNet, self).__init__()
        # no of instances for segmentation
        self.no_of_instances = 3  # if you want to output RGB instance map, it should be 3.
        self.use_temporal = use_temporal
        self.sequence_length = sequence_length
        print("Use {} as backbone".format(arch))
        if use_temporal:
            print("Temporal mode enabled with sequence_length={}".format(sequence_length))
        self._arch = arch
        if self._arch == 'UNet':
            self._encoder = UNet_Encoder(in_ch)
            self._encoder.to(DEVICE)

            self._decoder_binary = UNet_Decoder(2)
            self._decoder_instance = UNet_Decoder(self.no_of_instances)
            self._decoder_binary.to(DEVICE)
            self._decoder_instance.to(DEVICE)
        elif self._arch == 'ENet':
            self._encoder = ENet_Encoder(in_ch)
            self._encoder.to(DEVICE)
            
            # Freeze encoder when using temporal mode
            if self.use_temporal:
                for param in self._encoder.parameters():
                    param.requires_grad = False
                print("ENet encoder frozen for temporal training")
                
                # Initialize ConvLSTM for temporal processing
                # ENet encoder output: 128 channels, spatial size reduced by 8x
                encoder_output_channels = 128
                self.conv_lstm = ConvLSTM(
                    input_dim=encoder_output_channels,
                    hidden_dim=encoder_output_channels,
                    kernel_size=3
                )
                self.conv_lstm.to(DEVICE)
                print("ConvLSTM initialized for temporal processing")

            self._decoder_binary = ENet_Decoder(2)
            self._decoder_instance = ENet_Decoder(self.no_of_instances)
            self._decoder_binary.to(DEVICE)
            self._decoder_instance.to(DEVICE)
        elif self._arch == 'DeepLabv3+':
            self._encoder = Deeplabv3plus_Encoder()
            self._encoder.to(DEVICE)

            self._decoder_binary = Deeplabv3plus_Decoder(2)
            self._decoder_instance = Deeplabv3plus_Decoder(self.no_of_instances)
            self._decoder_binary.to(DEVICE)
            self._decoder_instance.to(DEVICE)
        else:
            raise("Please select right model.")

        self.relu = nn.ReLU().to(DEVICE)
        self.sigmoid = nn.Sigmoid().to(DEVICE)

    def forward(self, input_tensor):
        if self._arch == 'UNet':
            c1, c2, c3, c4, c5 = self._encoder(input_tensor)
            binary = self._decoder_binary(c1, c2, c3, c4, c5)
            instance = self._decoder_instance(c1, c2, c3, c4, c5)
        elif self._arch == 'ENet':
            if self.use_temporal:
                # Temporal mode: input shape should be [B, T, 3, H, W]
                # Handle both [B, T, 3, H, W] and [B, T*3, H, W] formats
                if input_tensor.dim() == 5:
                    # Already in [B, T, 3, H, W] format
                    B, T, C, H, W = input_tensor.shape
                elif input_tensor.dim() == 4 and input_tensor.shape[1] == self.sequence_length * 3:
                    # Reshape from [B, T*3, H, W] to [B, T, 3, H, W]
                    B, C_total, H, W = input_tensor.shape
                    T = self.sequence_length
                    input_tensor = input_tensor.view(B, T, 3, H, W)
                    B, T, C, H, W = input_tensor.shape
                else:
                    raise ValueError(f"Unexpected input shape for temporal mode: {input_tensor.shape}. Expected [B, T, 3, H, W] or [B, T*3, H, W]")
                
                # Process each frame through encoder
                # Encoder is frozen (requires_grad=False), so no gradients for encoder params
                # but gradients will still flow through activations to ConvLSTM and decoder
                encoded_frames = []
                for t in range(T):
                    frame = input_tensor[:, t]  # [B, 3, H, W]
                    # Ensure frame is float type (required for gradients)
                    if not frame.is_floating_point():
                        frame = frame.float()
                    # Ensure frame is on correct device
                    if frame.device != input_tensor.device:
                        frame = frame.to(input_tensor.device)
                    # Process through encoder - activations will have gradients even if encoder params don't
                    encoded = self._encoder(frame)  # [B, 128, H/8, W/8]
                    encoded_frames.append(encoded)
                
                # Stack encoded frames: [B, T, 128, H/8, W/8]
                encoded_sequence = torch.stack(encoded_frames, dim=1)
                
                # Process through ConvLSTM
                # ConvLSTM expects [B, T, C, H, W] and returns [B, C, H, W]
                c = self.conv_lstm(encoded_sequence)  # [B, 128, H/8, W/8]
                
                # Decode
                binary = self._decoder_binary(c)
                instance = self._decoder_instance(c)
            else:
                # Single frame mode: normal behavior
                c = self._encoder(input_tensor)
                binary = self._decoder_binary(c)
                instance = self._decoder_instance(c)
        elif self._arch == 'DeepLabv3+':
            c1, c2 = self._encoder(input_tensor)
            binary = self._decoder_binary(c1, c2)
            instance = self._decoder_instance(c1, c2)
        else:
            raise("Please select right model.")

        binary_seg_ret = torch.argmax(F.softmax(binary, dim=1), dim=1, keepdim=True)

        pix_embedding = self.sigmoid(instance)

        return {
            'instance_seg_logits': pix_embedding,
            'binary_seg_pred': binary_seg_ret,
            'binary_seg_logits': binary
        }