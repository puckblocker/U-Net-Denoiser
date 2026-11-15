# ========================================
# PURPOSE: U-NET DATA PREPROCESING
# ========================================

# LIBRARIES
import OpenEXR
import argparse
import os
import glob
import numpy as np
import random
import math
import Imath

import torch
import torch.nn as nn
import torch.nn.functional as f

def main():
    # FILE GRABBER 
    # USER INSTRUCTIONS / FILE LOCATIONS
    parser = argparse.ArgumentParser("U-Net Dataset Trainer")

    parser.add_argument('--dataset_dir', type=str, required=True, help="Path to EXR dataset")
    parser.add_argument('--output_dir', type=str, required=True, help="Path to denoised output")

    # Parse Arguments
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading input EXR files from: {args.dataset_dir}")
    print(f"Saving outputs to: {args.output_dir}")

    # SEARCH FOR EXR FILES & STORE DIR PATH
    filePath = os.path.join(args.dataset_dir, "**", "*.exr")
    files = sorted(glob.glob(filePath, recursive=True))

    if len(files) == 0:
        print("Error: No EXR files found in selected folder.")
        return
    print(f"Found {(files)}.")

    # HARDWARE SETUP
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')   # use cuda, if not use cpu
    print(f"Training with: {device}")

    # PREPOCESS DATA & MOVE DATA TO GPU
    xInput, yInput = GrabPatches(files)
    xTrain = torch.from_numpy(xInput).float().to(device)
    yTrain = torch.from_numpy(yInput).float().to(device)
    yTrain = yTrain[:, :3, :, :]

    # SETUP U-NET
    model = UNet(inChannels=16, outChannels=3).to(device)   # actual model
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    print(f"Training Starting...")

    # TRAINING LOOP
    epochs = 200
    for epoch in range(epochs):
        # RESET OPTIMIZER
        optimizer.zero_grad()

        # GRAB PREDICTIONS
        predictions = model(xTrain)

        # CALCULATE LOSS
        lossMSE = nn.MSELoss()(predictions, yTrain)
        lossSSIM = ssimFunction(predictions, yTrain)
        lossSobel = sobelFunction(predictions, yTrain)

        ## CALCULATE TOTAL LOSS
        loss = lossMSE + lossSSIM + lossSobel

        # BACKPROPOGATION
        loss.backward() # calculate what went wrong
        optimizer.step() # changes weights

        # LOSS REPORT
        if epoch % 10 == 0:
            psnr = psnrFunction(lossMSE, yTrain)
            print(f"Epoch: {epoch} | Total Loss: {loss.item():.5f} | Current PSNR: {psnr.item():.2f}")

            # SAVE CURRENT PATCH
            ## SAVE WEIGHTS
            weightPath = os.path.join(args.output_dir, f"unet_weight_epoch_{epoch}.pth")
            torch.save(model.state_dict(), weightPath)

            ## SAVE PATCH IMAGE
            crntImg = predictions.detach().cpu().numpy() # pull off gpu

            ## UNDO PREPOCESSING
            patch = crntImg[0]  # isolate patch
            patch = np.transpose(patch, (1, 2, 0))  # fix order

            patch = np.exp(patch) - 1.0
            patch = patch.astype(np.float32)

            # WRITE NEW EXR
            outPath = os.path.join(args.output_dir, f"test_patch_epoch_{epoch}.exr")
            header = OpenEXR.Header(patch.shape[1], patch.shape[0])
            chanType = Imath.Channel(Imath.PixelType(Imath.PixelType.FLOAT))
            header['channels'] = {'R': chanType, 'G': chanType, 'B': chanType}

            writer = OpenEXR.OutputFile(outPath, header)
            writer.writePixels({
                'R': patch[:, :, 0].tobytes(),
                'G': patch[:, :, 1].tobytes(),
                'B': patch[:, :, 2].tobytes(),
            })
            writer.close()

# ----------------------------------------
# READ IN EXR CHANNELS
# ----------------------------------------
def ReadEXR(file):
    with OpenEXR.File(file) as inFile:
        # GRAB AND STORE CHANNELS
        channelDict = inFile.channels()
        noisyRGB = channelDict["RGB"].pixels
        otherTargets = ["albedo", "albedoVariance.Z", "colorVariance.Z", "depth.Z", "depthVariance.Z", 
                        "normal", "normalVariance.Z", "visibility.Z", "visibilityVariance.Z"]
        otherBuffers = []

        try:
            for name in otherTargets:
                data = channelDict[name].pixels

                # RESHAPE DATA
                if len(data.shape) == 2:
                    data = np.expand_dims(data, axis=-1)

                otherBuffers.append(data)
        except:
            print(f"Warning: {os.path.basename(file)} missing {name}.")
            return None

    # ----------------------------------------
    # PREPROCESS VALUES
    # ----------------------------------------
    cleanRGB = np.nan_to_num(noisyRGB, nan=0.0) 
    cleanRGB = np.log(1 + cleanRGB)

    # COMBINE & CLEAN BUFFERS
    gBuffers = np.concatenate([cleanRGB] + otherBuffers, axis=-1)
    gBuffers = np.nan_to_num(gBuffers, nan=0.0)

    print(gBuffers.shape)

    return gBuffers

def GrabPatches(files):
    truthMap = {}
    noisyFiles = []
    finalNoisyPatches = []
    finalTruthPatches = []

    # NAME SEPARATOR
    for f in files:
        # PROCESS SCENE NAME FROM ORIGNAL NAME
        basename = os.path.basename(f)
        sceneName = basename.rsplit('-', 1)[0]

        if "08192spp.exr" in basename:
            truthMap[sceneName] = f
        else:
            noisyFiles.append((f))

    truthCache = {}

    # PROCESS NOISY FILES
    for inFile in noisyFiles:
        basename = os.path.basename(inFile)

        # SCENE LOOKUP
        crntScene = None
        for scene in truthMap.keys():
            if scene in basename:
                crntScene = scene
                break

        if crntScene is None:
            print(f"No ground truth found for {basename}")
            continue

        # READ EXR
        print(f"Processing noisy image: {basename}")
        fullNoisyBuffer = ReadEXR(inFile)

        # CHECK FOR MISSING VARIABLE
        if fullNoisyBuffer is None:
            continue

        # LOAD TRUTH INTO MEMORY CACHE
        if crntScene not in truthCache:
            print(f"Loading Ground Truth For {crntScene} Into RAM")
            truthBuffer = ReadEXR(truthMap[crntScene])

            # CHECK FOR MISSING VARIABLE
            if truthBuffer is None:
                print(f"Ground Truth File: {crntScene} is corrupted. Skipping")
                continue
            truthCache[crntScene] = truthBuffer

        # GRAB PATCH FOR NOISY AND GROUND TRUTH
        fullTruthBuffer = truthCache[crntScene]
        randY = random.randint(0,656)
        randX = random.randint(0,1226)

        noisyPatch = fullNoisyBuffer[randY: randY + 64, randX: randX + 64]
        truthPatch = fullTruthBuffer[randY: randY + 64, randX: randX + 64]

        finalNoisyPatches.append(noisyPatch)
        finalTruthPatches.append(truthPatch)

    # CREATE TENSORS (B, C, H, W) *16 Channels
    xTrain = np.stack(finalNoisyPatches, axis=0)
    yTrain = np.stack(finalTruthPatches, axis=0)
    xTrain = np.transpose(xTrain, (0, 3, 1, 2))
    yTrain = np.transpose(yTrain, (0, 3, 1, 2))

    print(f"Final Truth Shape: {yTrain}")
    print(f"Final Noisy Shape: {xTrain}")

    return xTrain, yTrain

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

# ----------------------------------------
# SSIM LOSS FUNCTION
# ----------------------------------------
def ssimFunction(x, y):
    C= x.shape[1]
    windowSize = 11
    L = torch.max(y)

    # GAUSSIAN BELL CURVE
    gauss = 1.5
    gauss1D = torch.Tensor([math.exp(-(i - windowSize//2)**2 / float(2 * gauss**2)) for i in range(windowSize)]) 
    gauss1D = gauss1D / gauss1D.sum() # normalize
    gauss2D = gauss1D.unsqueeze(1) * gauss1D.unsqueeze(0) # expand into grid

    gaussWeights = gauss2D.expand(C, 1, windowSize, windowSize).contiguous().to(x.device)# create 3 clone bell curves for rgb

    # MEANS
    averager = nn.Conv2d(in_channels=C, out_channels=C, kernel_size=windowSize, groups=C, bias=False)
    averager.weight.data = gaussWeights     # prevent random noise, instead use bell curve
    averager.weight.requires_grad = False   # prevent weights from altering

    meanX = averager(x)
    meanY = averager(y)

    # VARIANCES
    meanSqX = averager(x * x)
    sqrMeanX = meanX * meanX
    meanSqY = averager(y * y)
    sqrMeanY = meanY * meanY

    sigmaX = meanSqX - sqrMeanX 
    sigmaY = meanSqY - sqrMeanY

    # COVARIANCE
    meanSqXY = averager(x * y)
    sigmaCov = meanSqXY - (meanX * meanY)

    C1 = (0.01 * L) * (0.01 * L)
    C2 = (0.03 * L) * (0.03 * L)
    ssimNum = (2 * meanX * meanY + C1) * (2* sigmaCov + C2)
    ssimDet = ((meanX * meanX) + (meanY * meanY) + C1) * (sigmaX + sigmaY + C2)

    return 1.0 - torch.mean(ssimNum / ssimDet)

# ----------------------------------------
# PSNR EVALUATION FUNCTION
# ----------------------------------------
def psnrFunction(MSE, y):
    L = torch.max(y)
    return 20 * torch.log10(L / torch.sqrt(MSE))

# ----------------------------------------
# EDGE AWARE LOSS FUNCTION (SOBEL)
# ----------------------------------------
def sobelFunction(x, y):
    # SETUP
    C = x.shape[1]
    windowSize = 3

    # BUILD SOBEL OPERATOR
    sobelOp = torch.tensor([[-1.0, 0.0, 1.0],
                               [-2.0, 0.0, 2.0],
                               [-1.0, 0.0, 1.0]])
    sobelWeights = sobelOp.expand(C, 1, windowSize, windowSize).contiguous().to(x.device)

    # CREATE EDGE FILTER
    sobelFilter = nn.Conv2d(in_channels= C, out_channels= C, kernel_size=windowSize, groups=C, bias=False)
    sobelFilter.weight.data = sobelWeights
    sobelFilter.weight.requires_grad = False

    # COMPARE EDGES
    edgeX = sobelFilter(x)
    edgeY = sobelFilter(y)

    # GRAB ERROR
    return nn.MSELoss()(edgeX, edgeY)


if __name__ == "__main__":
    main()