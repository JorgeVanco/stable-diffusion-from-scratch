import torch
import numpy as np
from tqdm import tqdm
from ddpm import DDPMSampler

WIDTH = 512
HEIGHT = 512
LATENTS_WIDTH = WIDTH // 8
LATENTS_HEIGHT = HEIGHT // 8


def generate(
    prompt: str,
    uncond_prompt: str,  # Negative prompt or empty string
    input_image: torch.Tensor | None = None,
    strength: float = 0.8,
    do_cfg: bool = True,  # Whether to apply classifier-free guidance
    cfg_scale: float = 7.5,  # Scale for classifier-free guidance
    sampler_name: str = "ddpm",  # Name of the sampler to use
    n_inference_steps: int = 50,
    models: dict[str, torch.nn.Module] | None = None,
    seed: int | None = None,
    device: str | None = None,
    idle_device: str | None = None,
    tokenizer=None,
) -> torch.Tensor:
    """
    Generate an image from a text prompt using a diffusion model.
    Args:
        prompt (str): The text prompt to guide the image generation.
        uncond_prompt (str): The unconditional text prompt for guidance.
        input_image (torch.Tensor | None): An optional input image tensor.
        strength (float): The strength of the input image guidance.
        do_cfg (bool): Whether to apply classifier-free guidance.
        cfg_scale (float): The scale for classifier-free guidance.
        sampler_name (str): The name of the sampler to use.
        n_inference_steps (int): The number of inference steps.
        models (dict[str, torch.nn.Module] | None): The models to use for generation.
        seed (int | None): The random seed for generation.
        device (str | None): The device to run the model on.
        idle_device (str | None): The device to use for idle tensors.
        tokenizer: The tokenizer to use for text encoding.
    """
    with torch.no_grad():
        if not (0 < strength <= 1):
            raise ValueError("Strength must be in the range (0, 1]")

        if idle_device:
            to_idle = lambda x: x.to(idle_device)
        else:
            to_idle = lambda x: x

        generator = torch.Generator(device=device)
        if seed is None:
            generator.seed()
        else:
            generator.manual_seed(seed)

        clip = models["clip"]
        clip = clip.to(device)

        if do_cfg:
            # Convert prompts to text embeddings
            cond_tokens = tokenizer.batch_encode_plus(
                [prompt],
                padding="max_length",
                max_length=77,
                truncation=True,
            ).input_ids
            uncond_tokens = tokenizer.batch_encode_plus(
                [uncond_prompt],
                padding="max_length",
                max_length=77,
                truncation=True,
            ).input_ids

            # (Batch_size, Sequence_length) -> (Batch_size, Sequence_length)
            cond_tokens = torch.tensor(cond_tokens, dtype=torch.long, device=device)
            uncond_tokens = torch.tensor(
                uncond_tokens,
                dtype=torch.long,
                device=device,
            )

            # (Batch_size, Sequence_length) -> (Batch_size, Sequence_length, Embedding_dim)
            cond_context = clip(cond_tokens)
            uncond_context = clip(uncond_tokens)

            # (2, Sequence_length, Embedding_dim) = (2, 77, 768)
            context = torch.cat([cond_context, uncond_context], dim=0)
        else:
            # Convert it to list of tokens
            tokens = tokenizer.batch_encode_plus(
                [prompt],
                padding="max_length",
                max_length=77,
            ).input_ids
            tokens = torch.tensor(tokens, dtype=torch.long, device=device)

            # (Batch_size, Sequence_length) -> (Batch_size, Sequence_length, Embedding_dim) = (1, 77, 768)
            context = clip(tokens)

        clip = to_idle(clip)

        if sampler_name == "ddpm":
            sampler = DDPMSampler(generator)
            sampler.set_inference_steps(n_inference_steps)
        else:
            raise ValueError(f"Unknown sampler: {sampler_name}")

        latents_shape = (1, 4, LATENTS_HEIGHT, LATENTS_WIDTH)

        if input_image is not None:
            encoder = models["encoder"]
            encoder = encoder.to(device)

            input_image_tensor = input_image.resize((WIDTH, HEIGHT))
            input_image_tensor = torch.tensor(
                input_image_tensor, dtype=torch.float32, device=device
            )

            input_image_tensor = rescale(input_image_tensor, (0, 255), (-1, 1))

            # (Height, Width, Channels) -> (Batch_size, Height, Width, Channels)
            input_image_tensor = input_image_tensor.unsqueeze(0)
            # (Batch_size, Height, Width, Channels) -> (Batch_size, Channels, Height, Width)
            input_image_tensor = input_image_tensor.permute(0, 3, 1, 2)

            encoder_noise = torch.randn(
                latents_shape, dtype=torch.float32, device=device, generator=generator
            )
            # Run the image through the encoder of the VAE
            latents = encoder(input_image_tensor, encoder_noise)

            sampler.set_strength(strength)
            latents = sampler.add_noise(latents, sampler.timesteps[0])

            sampler = to_idle(sampler)

        else:
            # If we are doing text-to-image generation, we need to sample random noise from N(0, I)
            latents = torch.randn(
                latents_shape, dtype=torch.float32, device=device, generator=generator
            )

        diffusion = models["diffusion"]
        diffusion = diffusion.to(device)

        timesteps = tqdm(sampler.timesteps, desc="Sampling", unit="step")
        for i, t in enumerate(timesteps):
            time_embedding = get_time_embedding(t).to(device)

            model_input = latents.clone()
            if do_cfg:
                # (Batch_size, Channels, Height, Width) -> (Batch_size * 2, Channels, Height, Width)
                model_input = model_input.repeat(2, 1, 1, 1)

            # model_output is the predicted noise by the UNet
            model_output: torch.Tensor = diffusion(model_input, context, time_embedding)

            if do_cfg:
                output_cond, output_uncond = model_output.chunk(2, dim=0)
                model_output = cfg_scale * (output_cond - output_uncond) + output_uncond

            # Remove noise predicted by the UNet
            latents = sampler.step(t, latents, model_output)

        diffusion = to_idle(diffusion)

        decoder = models["decoder"]
        decoder = decoder.to(device)

        images = decoder(latents)
        decoder = to_idle(decoder)

        images = rescale(images, (-1, 1), (0, 255), clamp=True)
        images = images.permute(0, 2, 3, 1)  # (Batch_size, Height, Width, Channels)
        images = images.to("cpu", torch.uint8).numpy()

    return images[0]


def rescale(
    tensor: torch.Tensor,
    old_range: tuple[float, float],
    new_range: tuple[float, float],
    clamp: bool = False,
) -> torch.Tensor:
    """Rescale a tensor from old_range to new_range."""
    old_min, old_max = old_range
    new_min, new_max = new_range

    # Scale the tensor to [0, 1]
    scaled_tensor = (tensor - old_min) / (old_max - old_min)

    # Scale to the new range
    rescaled_tensor = scaled_tensor * (new_max - new_min) + new_min

    if clamp:
        rescaled_tensor = rescaled_tensor.clamp(new_min, new_max)

    return rescaled_tensor


def get_time_embedding(timestep: torch.Tensor) -> torch.Tensor:
    """
    Get the time embedding for a given timestep t.
    Args:
        timestep (torch.Tensor): The timestep tensor.
    Returns:
        torch.Tensor: The time embedding tensor.
    """
    half_dim = 160
    emb = torch.exp(
        -np.log(10000) * torch.arange(0, half_dim, dtype=torch.float32) / half_dim
    )
    emb = torch.tensor([timestep], dtype=torch.float32)[:, None] * emb[None, :]
    emb = torch.cat([emb.cos(), emb.sin()], dim=-1)
    return emb
