#!/usr/bin/env python3
"""
Metrics Interpreter for SimVP Velocity Prediction Training
Shows normalized metrics converted to physical velocity units
"""

import sys
import numpy as np

def interpret_metrics(normalized_loss, normalized_mae, normalized_mse, scale_factor, ssim, psnr):
    """
    Convert normalized metrics to physical velocity units
    
    Args:
        normalized_loss: Loss in normalized space [-0.95, 0.95]
        normalized_mae: MAE in normalized space
        normalized_mse: MSE in normalized space
        scale_factor: Scale factor used for normalization (from dataset)
        ssim: Structural Similarity Index
        psnr: Peak Signal-to-Noise Ratio
    """
    
    print("\n" + "="*70)
    print("SimVP VELOCITY PREDICTION - METRICS INTERPRETATION")
    print("="*70 + "\n")
    
    # Denormalize to physical units (m/s)
    physical_loss = normalized_loss * scale_factor
    physical_mae = normalized_mae * scale_factor
    physical_rmse = np.sqrt(normalized_mse) * scale_factor
    
    print("📊 NORMALIZED METRICS (Training/Validation Space)")
    print("-" * 70)
    print(f"  Loss (MSE):     {normalized_loss:.6f}")
    print(f"  MAE:            {normalized_mae:.6f}")
    print(f"  MSE:            {normalized_mse:.6f}")
    print(f"  Scale factor:   {scale_factor:.4f} (maps ±{scale_factor*0.95:.2f} m/s → ±0.95)")
    
    print("\n📈 PHYSICAL METRICS (m/s - Actual Velocity Error)")
    print("-" * 70)
    print(f"  Loss (MSE):     {physical_loss:.6f} m/s²")
    print(f"  MAE:            {physical_mae:.6f} m/s  ← Average error per pixel")
    print(f"  RMSE:           {physical_rmse:.6f} m/s")
    
    print("\n✓ IMAGE QUALITY METRICS")
    print("-" * 70)
    print(f"  SSIM:           {ssim:.4f} (0-1, higher=better)")
    print(f"  PSNR:           {psnr:.2f} dB (higher=better)")
    
    print("\n🎯 INTERPRETATION")
    print("-" * 70)
    
    # Loss interpretation
    if physical_loss < 0.01:
        loss_quality = "🟢 EXCELLENT"
    elif physical_loss < 0.05:
        loss_quality = "🟢 VERY GOOD"
    elif physical_loss < 0.1:
        loss_quality = "🟡 GOOD"
    else:
        loss_quality = "🔴 NEEDS IMPROVEMENT"
    
    # MAE interpretation
    if physical_mae < 0.5:
        mae_quality = "🟢 EXCELLENT"
    elif physical_mae < 1.0:
        mae_quality = "🟢 VERY GOOD"
    elif physical_mae < 2.0:
        mae_quality = "🟡 GOOD"
    else:
        mae_quality = "🔴 NEEDS IMPROVEMENT"
    
    # SSIM interpretation
    if ssim > 0.75:
        ssim_quality = "🟢 EXCELLENT"
    elif ssim > 0.70:
        ssim_quality = "🟢 VERY GOOD"
    elif ssim > 0.60:
        ssim_quality = "🟡 GOOD"
    else:
        ssim_quality = "🔴 NEEDS IMPROVEMENT"
    
    # PSNR interpretation
    if psnr > 35:
        psnr_quality = "🟢 EXCELLENT"
    elif psnr > 30:
        psnr_quality = "🟢 VERY GOOD"
    elif psnr > 25:
        psnr_quality = "🟡 GOOD"
    else:
        psnr_quality = "🔴 NEEDS IMPROVEMENT"
    
    print(f"  Loss:           {loss_quality}")
    print(f"  MAE (Absolute):  {mae_quality}")
    print(f"  SSIM (Structure):{ssim_quality}")
    print(f"  PSNR (Quality):  {psnr_quality}")
    
    print("\n💡 WHAT THIS MEANS")
    print("-" * 70)
    if physical_mae < 0.5 and ssim > 0.75:
        print("  ✓ Model is learning velocity patterns very well!")
        print("  ✓ Predictions are accurate and structurally similar to ground truth")
        print("  ✓ Continue training - model is converging nicely")
    elif physical_mae < 1.0 and ssim > 0.70:
        print("  ✓ Model is learning well!")
        print("  ✓ Average error is reasonable for velocity prediction")
        print("  ~ Consider continuing training or tuning hyperparameters")
    else:
        print("  ⚠ Model might need more training or tuning")
        print("  → Try: increase epochs, adjust learning rate, or change model size")
    
    print("\n" + "="*70 + "\n")


# Example: Your current training output
if __name__ == "__main__":
    
    print("\n🔍 EXAMPLE: Your Epoch 3 Results")
    print("="*70)
    
    # Your envelope fold data statistics (from test output)
    scale_factor = 9.2473  # From data loader output
    
    # Your Epoch 3 metrics (with fixed metrics.py)
    normalized_loss = 0.0008
    normalized_mae = 0.0412  # After metrics fix
    normalized_mse = normalized_loss  # For velocity prediction
    ssim = 0.7673
    psnr = 38.1765
    
    print(f"Scale factor: {scale_factor:.4f}")
    print(f"Normalized Loss: {normalized_loss}")
    print(f"Normalized MAE: {normalized_mae}")
    print(f"SSIM: {ssim}")
    print(f"PSNR: {psnr}\n")
    
    interpret_metrics(
        normalized_loss=normalized_loss,
        normalized_mae=normalized_mae,
        normalized_mse=normalized_mse,
        scale_factor=scale_factor,
        ssim=ssim,
        psnr=psnr
    )
    
    # Comparison example
    print("\n📊 COMPARISON EXAMPLE: SimVP vs Your MiT-B1 Model")
    print("="*70)
    
    simvp_mae_physical = 0.0412 * 9.2473  # ~0.38 m/s
    your_model_mae = 0.42  # Hypothetical value from main.py
    
    improvement = (your_model_mae - simvp_mae_physical) / your_model_mae * 100
    
    print(f"\nSimVP MAE:        {simvp_mae_physical:.4f} m/s")
    print(f"Your MiT-B1 MAE:  {your_model_mae:.4f} m/s")
    print(f"Improvement:      {improvement:+.1f}%")
    print(f"\n{'✓ Your model is BETTER!' if improvement > 0 else '✗ SimVP baseline is better'}")
    print("="*70 + "\n")
