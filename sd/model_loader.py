import torch
import torch.nn as nn
from clip import CLIP
from encoder import VAE_Encoder
from decoder import VAE_Decoder
from diffusion import DiffusionModel
import model_converter


def preload_models_from_standard_weights(
    ckpt_path: str, device: str = "cpu", dtype: torch.dtype = torch.bfloat16
) -> dict[str, nn.Module | nn.Sequential]:
    state_dict = model_converter.load_from_standard_weights(ckpt_path, device=device)

    encoder = VAE_Encoder().to(device)
    encoder.load_state_dict(state_dict["encoder"], strict=True)
    encoder.to(dtype)
    encoder.compile()

    decoder = VAE_Decoder().to(device)
    decoder.load_state_dict(state_dict["decoder"], strict=True)
    decoder.to(dtype)
    decoder.compile()

    clip = CLIP().to(device)
    clip.load_state_dict(state_dict["clip"], strict=True)
    clip.to(dtype)
    clip.compile()

    diffusion = DiffusionModel().to(device)
    diffusion.load_state_dict(state_dict["diffusion"], strict=True)
    diffusion.to(dtype)
    diffusion.compile()

    return {
        "encoder": encoder,
        "decoder": decoder,
        "clip": clip,
        "diffusion": diffusion,
    }
