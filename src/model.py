import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super().__init__()
        padding = [(i - 1) // 2 for i in kernel_size]  # 'same' padding     
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.norm = nn.InstanceNorm2d(out_channels)
        self.act = nn.LeakyReLU()
        
    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        return x


class Decoder(nn.Module):
    
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super().__init__() 
        padding = [(i - 1) // 2 for i in kernel_size]  # 'same' padding 
        self.tconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.norm = nn.InstanceNorm2d(out_channels)
        self.act = nn.ReLU()
   
    def forward(self, x):
        x = self.tconv(x)
        x = self.norm(x)
        x = self.act(x)
        return x


class UNet(nn.Module):

    def __init__(self, kernel_size=(5,5), stride=(2,1), encoder_decoder_channels=None):
        super().__init__()
        if encoder_decoder_channels is None:
            encoder_channels = [(16,32), (32, 32), (32,32), (32, 64)]
            decoder_channels=[(64, 32), (64,32), (64,32), (64,16)]

        in_channels = encoder_channels[0][0]
        self.pre_conv = nn.Sequential(nn.Conv2d(1, in_channels, kernel_size=3, padding=1),
                                      nn.LeakyReLU(),
                                      nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
                                      nn.LeakyReLU()
                                     )

        self.encoders = nn.ModuleList([Encoder(m, n, kernel_size, stride) for m, n in encoder_channels])
        self.decoders = nn.ModuleList([Decoder(m, n, kernel_size, stride) for m, n in decoder_channels])

        out_channels = decoder_channels[-1][-1]
        self.post_conv= nn.Sequential(nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                                      nn.ReLU(),
                                      nn.Conv2d(out_channels, 1, kernel_size=1, padding=0),
                                      nn.Sigmoid()
                                     )

        self.base_freq = (stride[0])**len(encoder_channels)
        self.base_frame = (stride[1])**len(decoder_channels)

    def pre_pad(self, specs):
        _, _, nfreq, nframe = specs.shape
        dtype=specs.dtype
        device=specs.device

        # Calculate the spectrogram size suitable for Unet
        pad_freq = int(np.ceil((nfreq-1)/self.base_freq) * self.base_freq) - nfreq + 1
        pad_frame = int(np.ceil((nframe-1)/self.base_frame) * self.base_frame) - nframe + 1
        specs = F.pad(specs, (0, pad_frame, 0, pad_freq))
        return specs, nfreq, nframe

    def forward(self, specs):
        # Pre process
        x, nfreq, nframe = self.pre_pad(specs)
        x = self.pre_conv(x)

        # UNet
        outputs = []
        for i in range(len(self.encoders)):
            outputs.append(x)
            x = self.encoders[i](outputs[-1])

        x = self.decoders[0](x)
        for i in range(1,len(self.decoders)):
            x = self.decoders[i](torch.cat((x, outputs[-i]), dim=1))

        # Post process
        est_mask = self.post_conv(x)
        return est_mask[:, 0, :nfreq, :nframe]