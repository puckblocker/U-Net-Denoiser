# ========================================
# PURPOSE: LOSS FUNCTIONS
# ========================================
import torch
import torch.nn as nn
import math

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