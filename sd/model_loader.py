import torch.nn as nn
from clip import CLIP
from encoder import VAE_Encoder
from decoder import VAE_Decoder
from diffusion import DiffusionModel
import model_converter


def preload_models_from_standard_weights(
    ckpt_path: str, device: str = "cpu"
) -> dict[str, nn.Module | nn.Sequential]:
    state_dict = model_converter.load_from_standard_weights(ckpt_path, device=device)

    encoder = VAE_Encoder().to(device)
    encoder.load_state_dict(state_dict["encoder"], strict=True)

    decoder = VAE_Decoder().to(device)
    decoder.load_state_dict(state_dict["decoder"], strict=True)

    clip = CLIP().to(device)
    clip.load_state_dict(state_dict["clip"], strict=True)

    diffusion = DiffusionModel().to(device)
    diffusion.load_state_dict(state_dict["diffusion"], strict=True)

    return {
        "encoder": encoder,
        "decoder": decoder,
        "clip": clip,
        "diffusion": diffusion,
    }
