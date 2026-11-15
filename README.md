# U-Net Monte Carlo Denoiser

A PyTorch implementation of a 16-channel U-Net architecture designed to denoise Monte Carlo path-traced renders. 

To preserve sharp geometric details and structural integrity, this model trains using a custom composite loss function that combines:
- **MSE (Mean Squared Error)** for raw color accuracy.
- **SSIM (Structural Similarity Index)** for perceptual structure.
- **Sobel Edge-Aware Loss** for preserving crisp edges and boundaries.

## Prerequisites

You will need **Python 3.8+** and a CUDA-capable NVIDIA GPU for hardware-accelerated training. 

The primary dependencies are PyTorch, NumPy, and OpenEXR.

## Installation

1. **Clone the repository and create a virtual environment:**
   ```powershell
   git clone [https://github.com/yourusername/U-Net-Denoiser.git](https://github.com/yourusername/U-Net-Denoiser.git)
   cd U-Net-Denoiser
   python -m venv venv
   ```

2. **Activate the virtual environment:**
   ```powershell
   .\venv\Scripts\activate
   ```

## Install Dependencies

1. **Install PyTorch with CUDA support**
   ```powershell
   python -m pip install torch torchvision torchaudio --index-url "[https://download.pytorch.org/whl/cu121](https://download.pytorch.org/whl/cu121)"
   ```

2. **Install remaining dependencies**
   ```powershell
   pip install OpenEXR numpy
   ```

## Dataset Structure

For the program to function, the dataset must contain only `.exr` files, and the ground truth files must be labeled with `08192spp` (this target can be altered in the code). 

You can place everything inside a single parent folder (or subfolders); the code will recursively navigate through the directory to find the files. The scene name and sample count **must** be separated by a dash (`-`).

**Example files:** 
* `myscene-08192spp.exr` (Ground truth)
* `myscene-00128spp.exr` (Noisy input)

## Train Network

1. **To view the help menu:**
   ```powershell
   python unet_denoiser.py --help
   ```

2. **To train the network:**
   ```powershell
   python unet_denoiser.py --dataset_dir "C:\path\to\dataset" --output_dir "C:\path\to\output"
   ```
