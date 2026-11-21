# ========================================
# PURPOSE: U-Net Training
# ========================================

# LIBRARIES
import OpenEXR
import os
import numpy as np
import math
import Imath

import torch
import torch.nn as nn
from loss_functions import ssimFunction, psnrFunction, sobelFunction
from file_processor import GrabPatches, fileGrabber
from unet import UNet

# ----------------------------------------
# MAIN FUNCTION
# ----------------------------------------
def main():
    # GRAB FILE ADDRESSES
    files, args = fileGrabber()

    # HARDWARE SETUP
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')   # use cuda, if not use cpu
    print(f"Training with: {device}")

    # SETUP U-NET
    model = UNet(inChannels=16, outChannels=3).to(device)   # actual model
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    print(f"Training Starting...")

    # SEPARATE NOISY AND GROUND TRUTH FILES
    truthMap = {}
    noisyFiles = []

    ## NAME SEPARATOR
    for f in files:
        # PROCESS SCENE NAME FROM ORIGNAL NAME
        basename = os.path.basename(f)
        sceneName = basename.rsplit('-', 1)[0]

        if "08192spp.exr" in basename:
            truthMap[sceneName] = f
        else:
            noisyFiles.append((f))

    batchSize = 4
    batchTotal = math.ceil(len(noisyFiles) / batchSize)

    # TRAINING LOOP
    epochs = 200
    for epoch in range(epochs):
        # PROCESS FILES IN BATCHES
        for batchCnt in range(batchTotal):
            # PREPOCESS DATA & MOVE DATA TO GPU
            xInput, yInput = GrabPatches(truthMap, noisyFiles, batchSize, batchCnt)

            # CHECK FOR EMPTY STACK
            if xInput is None:
                continue

            xTrain = torch.from_numpy(xInput).float().to(device)
            yTrain = torch.from_numpy(yInput).float().to(device)
            yTrain = yTrain[:, :3, :, :]

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
        if epoch % 20 == 0:
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

if __name__ == "__main__":
    main()