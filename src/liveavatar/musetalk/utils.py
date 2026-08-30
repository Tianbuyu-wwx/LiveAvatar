import os

from .models.unet import PositionalEncoding, UNet
from .models.vae import VAE


def load_all_model(
    unet_model_path=os.path.join("models", "musetalkV15", "unet.pth"),
    vae_model_path="models/sd-vae-ft-mse",
    unet_config=os.path.join("models", "musetalkV15", "musetalk.json"),
    device=None,
):
    vae = VAE(model_path=vae_model_path)
    print(f"load unet model from {unet_model_path}")
    unet = UNet(
        unet_config=unet_config,
        model_path=unet_model_path,
        device=device,
    )
    pe = PositionalEncoding(d_model=384)
    return vae, unet, pe
