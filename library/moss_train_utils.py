# Scheduler adapted from OpenMOSS MOSS-Image, Apache-2.0; see LICENSE-MOSS-Image.
"""MOSS flow matching and FLUX.2 latent conversion."""

import torch
from diffusers import FlowMatchEulerDiscreteScheduler
from diffusers.configuration_utils import register_to_config


def training_sigmas(total=1000, shift=1.0, device="cpu"):
    if total < 1 or shift <= 0:
        raise ValueError("num_train_timesteps and training_shift must be positive")
    values = torch.linspace(1, shift / (total + shift - 1), total, device=device)
    return shift * values / (1 + (shift - 1) * values)


def flow_batch(latents, scheduler, min_timestep=0, max_timestep=None):
    total = scheduler.config.num_train_timesteps
    maximum = total if max_timestep is None else max_timestep
    if not 0 <= min_timestep < maximum <= total:
        raise ValueError("Require 0 <= min_timestep < max_timestep <= num_train_timesteps")
    grid = training_sigmas(total, scheduler.config.training_shift, latents.device).flip(0)
    indices = torch.randint(min_timestep, maximum, (latents.shape[0],), device=latents.device)
    sigmas = grid[indices]
    noise = torch.randn(latents.shape, device=latents.device, dtype=torch.float32)
    sigma = sigmas.view(-1, 1, 1, 1)
    clean = latents.float()
    return (1 - sigma) * clean + sigma * noise, noise - clean, sigmas * total


class MOSSImageScheduler(FlowMatchEulerDiscreteScheduler):
    @register_to_config
    def __init__(self, num_train_timesteps=1000, shift=5.0, training_shift=1.0):
        super().__init__(num_train_timesteps=num_train_timesteps, shift=1.0)
        self.register_to_config(shift=shift, training_shift=training_shift)

    def set_timesteps(self, num_inference_steps=50, device=None):
        if num_inference_steps < 1 or self.config.shift <= 0:
            raise ValueError("steps and shift must be positive")
        total = self.config.num_train_timesteps
        values = torch.linspace(1, self.config.shift / (total + self.config.shift - 1), num_inference_steps)
        sigmas = self.config.shift * values / (1 + (self.config.shift - 1) * values)
        grid = training_sigmas(total, self.config.training_shift).flip(0)
        upper = torch.searchsorted(grid, sigmas).clamp(max=total - 1)
        lower = (upper - 1).clamp(min=0)
        nearest = torch.where((sigmas - grid[lower]).abs() <= (grid[upper] - sigmas).abs(), lower, upper)
        self._shift = 1.0
        super().set_timesteps(device=device, sigmas=torch.unique_consecutive(grid[nearest]).tolist())


def decode_latents(vae, latents):
    z = latents.to(device=vae.device, dtype=vae.dtype)
    z = z * (vae.bn.running_var.float() + vae.config.batch_norm_eps).sqrt().to(z).view(1, -1, 1, 1)
    z = z + vae.bn.running_mean.to(z).view(1, -1, 1, 1)
    b, c, h, w = z.shape
    z = z.reshape(b, c // 4, 2, 2, h, w).permute(0, 1, 4, 2, 5, 3).reshape(b, c // 4, 2 * h, 2 * w)
    return vae.decode(z, return_dict=False)[0].clamp(-1, 1)
