# Load the SeapixGAN model globally when the application starts
import torch
import torch.nn as nn
import torch.nn.functional as F

class _DecodeLayer(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=False):
        super(_DecodeLayer, self).__init__()
        layers = [
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True)
        ]
        if dropout:
            layers.append(nn.Dropout(0.5))  # Dropout is used optionally
        self.decode = nn.Sequential(*layers)

    def forward(self, x, skip):
        # Perform upsampling and concatenate the skip connection from the encoder
        x = self.decode(x)
        return torch.cat((x, skip), dim=1)  # Concatenating along the channel dimension


class SeapixGANGenerator(nn.Module):
    def __init__(self):
        super(SeapixGANGenerator, self).__init__()

        # Encoder layers
        self.encoder = nn.ModuleList([
            self.conv_block(3, 64),
            self.conv_block(64, 128),
            self.conv_block(128, 256),
            self.conv_block(256, 512),
            self.conv_block(512, 512),
            self.conv_block(512, 512),
            self.conv_block(512, 512)
        ])
        
        # Decoder layers using the _DecodeLayer class
        self.decoder = nn.ModuleList([
            _DecodeLayer(512, 512, dropout=True),
            _DecodeLayer(1024, 512, dropout=True),
            _DecodeLayer(1024, 512, dropout=True),
            _DecodeLayer(1024, 256),
            _DecodeLayer(512, 128),
            _DecodeLayer(256, 64)
        ])
        
        # Final output layer
        self.final = nn.Sequential(
            nn.ConvTranspose2d(128, 3, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )
    
    def conv_block(self, in_channels, out_channels, kernel_size=4, stride=2, padding=1, dropout=False):
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
            nn.InstanceNorm2d(out_channels),  # Use InstanceNorm instead of BatchNorm
            nn.LeakyReLU(0.2, inplace=True)
        ]
        if dropout:
            layers.append(nn.Dropout(0.5))
        return nn.Sequential(*layers)

    def forward(self, x):
        enc_outputs = []

        # Encoder path
        for layer in self.encoder:
            x = layer(x)
            enc_outputs.append(x)

        # Decoder path with skip connections
        for i, layer in enumerate(self.decoder):
            x = layer(x, enc_outputs[-i-2])  # Use _DecodeLayer for decoding with skip connections

        # Final layer
        return self.final(x)