# GPTCast – Evaluation & Testing Notebook
# - Trained VAE and a GPTCast 8x8 model paths: 

#%% 1. Imports & setup

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# GPTCast imports (adjust if your paths differ)
from gptcast.models.autoencoder import AutoencoderKL
from gptcast.models.gptcast import GPTCast
from gptcast.datasets.mwae import MWAEDataset

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

base_path = "/home/jovyan/nfs/cemauriello/GPTCastNew/"
VAE_CKPT_OG = Path(f"{base_path}/models/vae_mae.ckpt")
GPT8_CKPT_OG = Path(f"{base_path}/models/gptcast_8.ckpt")
#VAE_CKPT_SK = Path(f"{base_path}/models_skipped/vae_mae.ckpt")
#GPT8_CKPT_SK = Path(f"{base_path}/models_skipped/gptcast_8x8.ckpt")
DATA_ROOT = Path(f"{base_path}/data")

#%% 2. Load dataset

# This mirrors the test dataset usage in the original notebooks

test_ds = MWAEDataset(
    root=DATA_ROOT,
    split="test",
    context_frames=4,
    target_frames=12,
)

# Single sample loader for qualitative tests
def get_sample(idx=0):
    sample = test_ds[idx]
    x = sample["context"].unsqueeze(0).to(DEVICE)
    y = sample["target"].unsqueeze(0).to(DEVICE)
    return x, y

#%% 3. Load trained VAE

vae = AutoencoderKL.load_from_checkpoint(VAE_CKPT)
vae = vae.to(DEVICE)
vae.eval()

# ----------------------
# VAE reconstruction test
# ----------------------

x, y = get_sample(0)

with torch.no_grad():
    z = vae.encode(x).sample()
    x_rec = vae.decode(z)

# Visualization helper

def show_frames(frames, title, n=4):
    plt.figure(figsize=(12, 3))
    for i in range(n):
        plt.subplot(1, n, i+1)
        plt.imshow(frames[0, i].cpu(), cmap="turbo")
        plt.axis("off")
    plt.suptitle(title)
    plt.show()

show_frames(x, "Original Context")
show_frames(x_rec, "VAE Reconstruction")

#%% 4. Load GPTCast 8x8

gpt = GPTCast.load_from_checkpoint(
    GPT_CKPT,
    first_stage_model=vae
)

gpt = gpt.to(DEVICE)
gpt.eval()

#%% 5. Autoregressive forecasting test

with torch.no_grad():
    forecast = gpt.sample(
        conditioning=x,
        steps=y.shape[1]
    )

show_frames(y, "Ground Truth Future")
show_frames(forecast, "GPTCast Forecast")

#%% 6. Quantitative metrics 

from torchmetrics.functional import mean_squared_error

mse = mean_squared_error(forecast, y)
print(f"MSE: {mse.item():.6f}")

# ----------------------
# CRPS (Continuous Ranked Probability Score)
# ----------------------

def crps_deterministic(pred, gt):
    """
    pred, gt: tensors of shape [B, T, H, W]
    Returns mean CRPS over all dimensions
    """
    return torch.mean(torch.abs(pred - gt))

crps_val = crps_deterministic(forecast, y)
print(f"CRPS: {crps_val.item():.6f}")

# ----------------------
# Thresholded precipitation score (IoU-style)
# ----------------------

def threshold_score(pred, gt, thr=0.1):
    pred_bin = (pred > thr).float()
    gt_bin = (gt > thr).float()
    intersection = (pred_bin * gt_bin).sum()
    union = pred_bin.sum() + gt_bin.sum() - intersection
    return (intersection / (union + 1e-6)).item()

print("IoU@thr=0.1:", threshold_score(forecast, y))

# ======================

from torchmetrics.functional import mean_squared_error

mse = mean_squared_error(forecast, y)
print(f"MSE: {mse.item():.6f}")

# Optional: thresholded precipitation score example

def threshold_score(pred, gt, thr=0.1):
    pred_bin = (pred > thr).float()
    gt_bin = (gt > thr).float()
    intersection = (pred_bin * gt_bin).sum()
    union = pred_bin.sum() + gt_bin.sum() - intersection
    return (intersection / (union + 1e-6)).item()

print("IoU@thr=0.1:", threshold_score(forecast, y))

#%% 7. Batch evaluation loop 

from torch.utils.data import DataLoader

test_loader = DataLoader(test_ds, batch_size=4, shuffle=False)

mse_vals = []

with torch.no_grad():
    for batch in test_loader:
        x = batch["context"].to(DEVICE)
        y = batch["target"].to(DEVICE)
        pred = gpt.sample(conditioning=x, steps=y.shape[1])
        mse_vals.append(mean_squared_error(pred, y).item())

print(f"Average MSE over test set: {np.mean(mse_vals):.6f}")

#%% Notes
# - This notebook mirrors the *evaluation logic* of the original GPTCast notebooks
# - You can now directly compare:
#   • Original decoder vs modified decoder
#   • VAE reconstruction quality
#   • Forecast sharpness & stability
# - For publication-style metrics (CSI, FSS), add thresholded variants here
