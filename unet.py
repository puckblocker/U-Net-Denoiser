# ========================================
# PURPOSE: U-NET
# ========================================
import torch
import torch.nn as nn

# ----------------------------------------
# U-NET HELPER
# ----------------------------------------
class DoubleConvBlock(nn.Module):
    def __init__(self, inChannels, outChannels):
        super(DoubleConvBlock, self).__init__()

        # CHAIN MODULES (double conv to expand filter's fov)
        self.doubleConv = nn.Sequential(
            nn.Conv2d(inChannels, outChannels, kernel_size=3, padding=1),  # apply convolution across image
            nn.ReLU(inplace=True),  # clamp negs to 0
            nn.Conv2d(outChannels, outChannels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.doubleConv(x)

# ----------------------------------------
# U-NET DENOISER
# ----------------------------------------
class UNet(nn.Module):
    # ----------------------------------------
    # U-NET STRUCTURE
    # ----------------------------------------
    def __init__(self, inChannels, outChannels):
        # WAKE UP MODULE
        super(UNet, self).__init__()

        # ENCODER MEMORY ALLOCATION (LAYER 1-4)
        self.enc1 = DoubleConvBlock(inChannels=inChannels, outChannels=64)
        self.enc2 = DoubleConvBlock(inChannels=64, outChannels=128)
        self.enc3 = DoubleConvBlock(inChannels=128, outChannels=256)
        self.enc4 = DoubleConvBlock(inChannels=256, outChannels=512)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # BOTTLENECK MEMORY ALLOCATION
        self.bottleneck = DoubleConvBlock(inChannels=512, outChannels=1024)

        # DECODER MEMORY ALLOCATION
        self.upConv1 = nn.ConvTranspose2d(in_channels=1024, out_channels=512, kernel_size=2, stride=2)
        self.dec1 = DoubleConvBlock(inChannels=1024, outChannels=512)
        self.upConv2 = nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=2, stride=2)
        self.dec2 = DoubleConvBlock(inChannels=512, outChannels=256)
        self.upConv3 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=2, stride=2)
        self.dec3 = DoubleConvBlock(inChannels=256, outChannels=128)
        self.upConv4 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=2, stride=2)
        self.dec4 = DoubleConvBlock(inChannels=128, outChannels=64)

        # OUTPUT RGB MEMORY ALLOCATION
        self.finalConv = nn.Conv2d(in_channels=64, out_channels=outChannels, kernel_size=1)

    def forward(self, x):
        # ENCODER
        ## SKIP CONNECTIONS & SHRINK
        skip1 = self.enc1(x)
        shrink1 = self.pool(skip1)
        
        skip2 = self.enc2(shrink1)
        shrink2 = self.pool(skip2)
        
        skip3 = self.enc3(shrink2)
        shrink3 = self.pool(skip3)

        skip4 = self.enc4(shrink3)
        shrink4 = self.pool(skip4)

        # BOTTLENECK
        bottle = self.bottleneck(shrink4)

        # DECODER
        ## ENLARGE AND COMBINE
        up1 = self.upConv1(bottle)
        merge1 = torch.cat([up1, skip4], dim=1)
        dec1 = self.dec1(merge1)

        up2 = self.upConv2(dec1)
        merge2 = torch.cat([up2, skip3], dim=1)
        dec2 = self.dec2(merge2)

        up3 = self.upConv3(dec2)
        merge3 = torch.cat([up3, skip2], dim=1)
        dec3 = self.dec3(merge3)

        up4 = self.upConv4(dec3)
        merge4 = torch.cat([up4, skip1], dim=1)
        dec4 = self.dec4(merge4)

        ## GET OUTPUT IMAGE
        output = self.finalConv(dec4)

        return output