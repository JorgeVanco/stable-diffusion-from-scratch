import torch
import numpy as np


class DDPMSampler:
    """
    A class that implements the DDPM (Denoising Diffusion Probabilistic Models) sampler.
    This sampler is used to add noise to samples and perform inference steps in a diffusion model.
    """

    def __init__(
        self,
        generator: torch.Generator,
        num_training_steps: int = 1000,
        beta_start: float = 0.00085,
        beta_end: float = 0.0120,
    ) -> None:
        self.betas = (
            torch.linspace(
                beta_start**0.5, beta_end**0.5, num_training_steps, dtype=torch.float32
            )
            ** 2
        )
        self.alphas = 1.0 - self.betas
        self.alpha_cumprod = torch.cumprod(self.alphas, dim=0)
        self.one = torch.tensor(1.0, dtype=torch.float32)

        self.generator = generator

        self.num_training_steps: int = num_training_steps
        self.timesteps: torch.Tensor = torch.from_numpy(
            np.arange(0, num_training_steps)[::-1]
        ).long()
        self.num_inference_steps: int
        self.inference_timesteps: torch.Tensor
        self.start_step: int

    def set_inference_steps(self, num_inference_steps: int = 50) -> None:
        """
        Set the number of inference steps for the DDPM sampler.
        Args:
            num_inference_steps (int): The number of inference steps to set.
        Raises:
            ValueError: If num_inference_steps is greater than num_training_steps.
        """
        if num_inference_steps > self.num_training_steps:
            raise ValueError(
                f"num_inference_steps ({num_inference_steps}) cannot be greater than num_training_steps ({self.num_training_steps})"
            )
        self.num_inference_steps = num_inference_steps
        step_ratio = self.num_training_steps // num_inference_steps
        self.inference_timesteps = self.timesteps[::step_ratio]
        self.inference_timesteps = self.inference_timesteps.to(self.generator.device)

    def _get_previous_timestep(self, t: int) -> int:
        """
        Get the previous timestep for the given timestep t.
        Args:
            t (int): The current timestep.
        Returns:
            int: The previous timestep.
        """
        prev_t = t - self.num_training_steps // self.num_inference_steps
        return prev_t

    def _get_variance(self, t: int) -> torch.Tensor:
        """
        Compute the variance for the given timestep t.
        Args:
            t (int): The current timestep.
        Returns:
            torch.Tensor: The variance for the given timestep.
        """
        prev_t = self._get_previous_timestep(t)

        alpha_prod_t = self.alpha_cumprod[t]
        alpha_prod_t_prev = self.alpha_cumprod[prev_t] if prev_t >= 0 else self.one

        current_beta_t = 1 - alpha_prod_t / alpha_prod_t_prev

        # Compute the variance using the formula (7) from the DDPM paper
        variance = (1 - alpha_prod_t_prev) / (1 - alpha_prod_t) * current_beta_t

        variance = torch.clamp(variance, min=1e-20)  # Avoid division by zero
        return variance

    def set_strength(self, strength: float = 1.0) -> None:
        """
        Set the strength of the DDPM sampler.
        Args:
            strength (float): The strength of the DDPM sampler, where 1.0 means full strength.
        Raises:
            ValueError: If strength is not in the range [0, 1].
        """
        if not (0 <= strength <= 1):
            raise ValueError(f"strength must be in the range [0, 1], got {strength}")
        start_step = self.num_inference_steps - int(strength * self.num_inference_steps)
        self.timesteps = self.timesteps[start_step:]
        self.start_step = start_step

    def step(
        self, t: int, latents: torch.Tensor, model_output: torch.Tensor
    ) -> torch.FloatTensor:
        """
        Perform a single step of the DDPM sampler.
        Args:
            t (int): The current timestep.
            latents (torch.Tensor): The current latent samples.
            model_output (torch.Tensor): The output from the model at the current timestep.
        Returns:
            torch.FloatTensor: The predicted previous sample.
        """
        prev_t = self._get_previous_timestep(t)

        alpha_prod_t = self.alpha_cumprod[t]
        alpha_prod_t_prev = self.alpha_cumprod[prev_t] if prev_t >= 0 else self.one
        beta_prod_t = 1 - alpha_prod_t
        beta_prod_t_prev = 1 - alpha_prod_t_prev
        current_alpha_t = alpha_prod_t / alpha_prod_t_prev
        current_beta_t = 1 - current_alpha_t

        # Compute the predicted original sample using formula (15) from the DDPM paper
        pred_original_sample = (
            latents - beta_prod_t.sqrt() * model_output
        ) / alpha_prod_t.sqrt()

        # Compute the coefficients for the pred_original_sample and current sample x_t
        pred_original_sample_coeff = (
            alpha_prod_t.sqrt() * current_beta_t
        ) / beta_prod_t
        current_sample_coeff = current_alpha_t * beta_prod_t_prev / beta_prod_t

        # Compute the predicted previous sample mean
        pred_prev_sample = (
            pred_original_sample_coeff * pred_original_sample
            + current_sample_coeff * latents
        )

        std: torch.Tensor = torch.tensor(
            0.0, dtype=latents.dtype, device=latents.device
        )
        if t > 0:
            device = model_output.device
            noise = torch.randn(
                model_output.shape,
                generator=self.generator,
                device=device,
                dtype=model_output.dtype,
            )

            std = self._get_variance(t).sqrt() * noise

        # N(0, 1) --> N(mu, sigma^2)
        # X = mu + sigma * Z where Z ~ N(0, 1)
        pred_prev_sample = pred_prev_sample + std

        return pred_prev_sample

    def add_noise(
        self, original_samples: torch.Tensor, t: torch.IntTensor
    ) -> torch.FloatTensor:
        """
        Add noise to the original samples based on the given timestep t.
        Args:
            original_samples (torch.Tensor): The original samples to which noise will be added.
            t (torch.IntTensor): The timestep at which noise is added.
        Returns:
            torch.FloatTensor: The samples with added noise.
        """

        mean = self.alpha_cumprod[t].sqrt().flatten() * original_samples

        while len(mean.shape) < len(original_samples.shape):
            mean = mean.unsqueeze(-1)

        std = (1 - self.alpha_cumprod[t]).sqrt().flatten()
        while len(std.shape) < len(original_samples.shape):
            std = std.unsqueeze(-1)

        noise = torch.randn(
            original_samples.shape,
            generator=self.generator,
            device=original_samples.device,
        )

        return mean + std * noise
