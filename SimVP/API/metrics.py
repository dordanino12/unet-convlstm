import numpy as np
from skimage.metrics import structural_similarity as cal_ssim

def MAE(pred, true, mask=None):
    """Mean Absolute Error - averaged across all dimensions or masked"""
    if mask is not None:
        # Sum the error only where the mask is 1, and divide by the number of cloud pixels
        return np.sum(np.abs(pred - true) * mask) / np.maximum(np.sum(mask), 1e-8)
    return np.mean(np.abs(pred - true))

def MSE(pred, true, mask=None):
    """Mean Squared Error - averaged across all dimensions or masked"""
    if mask is not None:
        # Sum the squared error only where the mask is 1, and divide by the number of cloud pixels
        return np.sum(((pred - true) ** 2) * mask) / np.maximum(np.sum(mask), 1e-8)
    return np.mean((pred - true) ** 2)

# cite the `PSNR` code from E3d-LSTM, Thanks!
# https://github.com/google/e3d_lstm/blob/master/src/trainer.py line 39-40
def PSNR(pred, true):
    mse = np.mean((np.uint8(pred * 255)-np.uint8(true * 255))**2)
    return 20 * np.log10(255) - 10 * np.log10(mse)

def metric(pred, true, mean, std, return_ssim_psnr=False, clip_range=[0, 1], mask=None):
    pred = pred*std + mean
    true = true*std + mean
    
    # Calculate errors with the new mask support
    mae = MAE(pred, true, mask)
    mse = MSE(pred, true, mask)

    if return_ssim_psnr:
        pred = np.maximum(pred, clip_range[0])
        pred = np.minimum(pred, clip_range[1])
        ssim, psnr = 0, 0
        
        # Handle single-channel (velocity) vs multi-channel (RGB) data
        for b in range(pred.shape[0]):
            for f in range(pred.shape[1]):
                # For velocity prediction with duplicated channels, use only first channel
                if pred.shape[2] == 1:
                    # Single channel: compute SSIM without multichannel
                    ssim += cal_ssim(pred[b, f, 0], true[b, f, 0], data_range=clip_range[1] - clip_range[0])
                else:
                    # Multi-channel: Use first channel for SSIM (velocity is duplicated)
                    ssim += cal_ssim(pred[b, f, 0], true[b, f, 0], data_range=clip_range[1] - clip_range[0])
                psnr += PSNR(pred[b, f], true[b, f])
        ssim = ssim / (pred.shape[0] * pred.shape[1])
        psnr = psnr / (pred.shape[0] * pred.shape[1])
        return mse, mae, ssim, psnr
    else:
        return mse, mae