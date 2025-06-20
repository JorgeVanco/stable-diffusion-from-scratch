import torch
import numpy as np
from .ddpm import DDPMSampler

class DDIMSampler(DDPMSampler):
    def __init__(
        self,
        generator: torch.Generator,
        num_training_steps: int = 1000,
        beta_start: float = 0.00085,
        beta_end: float = 0.0120,
        ddim_eta: float = 0.0,
    ) -> None:
        super().__init__(generator, num_training_steps, beta_start, beta_end)
        self.ddim_eta = ddim_eta  # eta parameter for DDIM
        
        
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
        
        # Compute the variance using the formula of DDIM (Denoising Diffusion Implicit Models) paper
        variance = self.ddim_eta ** 2 * (1 - alpha_prod_t_prev) / (1 - alpha_prod_t) * current_beta_t

        return torch.clamp(variance, min=0.0)


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
        
        std_noise: torch.Tensor = torch.tensor(
            0.0, dtype=latents.dtype, device=latents.device
        )
        variance: torch.Tensor = self._get_variance(t)
        if t > 0 and variance > 0.0:
            device = model_output.device
            noise = torch.randn(
                model_output.shape,
                generator=self.generator,
                device=device,
                dtype=model_output.dtype,
            )

            std_noise = variance.sqrt() * noise
        
        pred_original_sample = (latents - beta_prod_t.sqrt() * model_output) / alpha_prod_t.sqrt()
        
        pred_original_sample_coeff = alpha_prod_t_prev.sqrt()
        current_sample_coeff = torch.sqrt(beta_prod_t_prev - variance)
        pred_prev_sample = pred_original_sample_coeff * pred_original_sample + current_sample_coeff * model_output + std_noise

        # # Compute the predicted original sample using formula (15) from the DDPM paper
        # pred_original_sample = (
        #     latents - beta_prod_t.sqrt() * model_output
        # ) / alpha_prod_t.sqrt()

        # # Compute the coefficients for the pred_original_sample and current sample x_t
        # pred_original_sample_coeff = (
        #     alpha_prod_t_prev.sqrt() * current_beta_t
        # ) / beta_prod_t
        # current_sample_coeff = current_alpha_t.sqrt() * beta_prod_t_prev / beta_prod_t

        # # Compute the predicted previous sample mean
        # pred_prev_sample = (
        #     pred_original_sample_coeff * pred_original_sample
        #     + current_sample_coeff * latents
        # )

        

        # N(0, 1) --> N(mu, sigma^2)
        # X = mu + sigma * Z where Z ~ N(0, 1)
        # pred_prev_sample = pred_prev_sample + std

        return pred_prev_sample