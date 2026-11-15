# U-Net-Denoiser
This is an implementation of a U-Net that utilizes SSIM, MSE, and an edge aware loss function (focuses on sobel operator) to denoiser Monte Carlo Path Traced Renders

# Help Menu
Run `python unet_denoiser.py -- help`

# How to Run
Have a specified dataset input folder and output folder
Run `python unet_denoiser.py --dataset_dir "<parent folder>" --output_dir "<output folder>"`
