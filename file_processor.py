# ========================================
# PURPOSE: FILE PROCESSING
# ========================================
import argparse
import os
import glob
import OpenEXR
import random
import numpy as np

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

def GrabPatches(truthMap, noisyFiles, batchSize, batchCnt):
    # INDEXING
    count = batchSize
    startIndx = batchSize * batchCnt
    endIndx = startIndx + batchSize

    truthCache = {}
    noisyFiles = noisyFiles[startIndx:endIndx]
    finalNoisyPatches = []
    finalTruthPatches = []

    # PROCESS NOISY FILES
    for inFile in noisyFiles:
        fileName = os.path.basename(inFile)
        crntScene = fileName.rsplit('-', 1)[0]

        # SCENE LOOKUP
        if crntScene not in truthMap:
            print(f"No ground truth found for {fileName}")
            continue

        # READ EXR
        print(f"Processing noisy image: {fileName}")
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
        patchSize = 256
        maxY = fullNoisyBuffer.shape[0]
        maxX = fullNoisyBuffer.shape[1]

        randY = random.randint(0,maxY - patchSize)
        randX = random.randint(0,maxX - patchSize)

        noisyPatch = fullNoisyBuffer[randY: randY + patchSize, randX: randX + patchSize]
        truthPatch = fullTruthBuffer[randY: randY + patchSize, randX: randX + patchSize]

        finalNoisyPatches.append(noisyPatch)
        finalTruthPatches.append(truthPatch)

        # UPDATE & CHECK COUNT
        if count <= 0:
            break

        count -= 1

    # CHECK FOR EMPTY STACK
    if len(finalNoisyPatches) == 0:
        print(f"Error: No files in noisy file batch.")
        return None, None

    # CREATE TENSORS (B, C, H, W) *16 Channels
    xTrain = np.stack(finalNoisyPatches, axis=0)
    yTrain = np.stack(finalTruthPatches, axis=0)
    xTrain = np.transpose(xTrain, (0, 3, 1, 2))
    yTrain = np.transpose(yTrain, (0, 3, 1, 2))

    print(f"Final Truth Shape: {yTrain.shape}")
    print(f"Final Noisy Shape: {xTrain.shape}")

    return xTrain, yTrain

# ----------------------------------------
# FILE GRABBER FUNCTION
# ----------------------------------------
def fileGrabber():
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
    print(f"Found {(len(files))} files.")

    return files, args