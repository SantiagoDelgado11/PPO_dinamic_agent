from __future__ import annotations

import torch


class StateBuilder:
    """Build bounded, deployment-safe state vectors for solver selection."""

    def __init__(self, args=None) -> None:
        self.eps = float(getattr(args, "eps", 1e-8))
        self.state_clip = float(getattr(args, "state_clip", 1.0))
        self.log_scale = float(getattr(args, "log_scale", 6.0))
        self.state_dim = 13

    def _signed_log1p(self, value: torch.Tensor) -> torch.Tensor:
        return torch.sign(value) * torch.log1p(value.abs())

    def _bounded_log_feature(self, value: torch.Tensor) -> torch.Tensor:
        scaled = self._signed_log1p(value) / self.log_scale
        return torch.clamp(scaled, -1.0, 1.0)

    def _normalize_action(self, previous_action: int, action_count: int, device: torch.device) -> torch.Tensor:
        if action_count <= 1:
            return torch.tensor(0.0, device=device)
        centered = (2.0 * float(previous_action) / float(action_count - 1)) - 1.0
        return torch.tensor(centered, dtype=torch.float32, device=device)

    def build(
        self,
        y: torch.Tensor,
        H,
        x_latent: torch.Tensor,
        x_estimate: torch.Tensor,
        previous_estimate: torch.Tensor | None,
        iteration: int,
        max_iterations: int,
        diffusion_timestep: int,
        max_diffusion_timestep: int,
        previous_action: int,
        action_count: int,
        previous_consistency: float,
    ) -> torch.Tensor:
        device = x_estimate.device
        eps = self.eps

        residual = H.forward_pass(x_estimate) - y
        fidelity = torch.mean(residual.square())
        gradient = H.transpose_pass(residual)
        grad_norm = torch.linalg.norm(gradient)

        x_energy = torch.mean(x_estimate.square())
        latent_energy = torch.mean(x_latent.square())
        measurement_energy = torch.mean(y.square())
        spatial_variance = torch.var(x_estimate, unbiased=False)

        if previous_estimate is None:
            convergence_ratio = torch.tensor(0.0, device=device)
            cosine_alignment = torch.tensor(1.0, device=device)
        else:
            delta = x_estimate - previous_estimate
            convergence_ratio = torch.linalg.norm(delta) / (torch.linalg.norm(x_estimate) + eps)
            cosine_alignment = torch.nn.functional.cosine_similarity(
                x_estimate.reshape(1, -1),
                previous_estimate.reshape(1, -1),
                dim=1,
                eps=eps,
            ).squeeze(0)

        residual_to_signal = fidelity / (measurement_energy + eps)
        normalized_iteration = torch.tensor(
            2.0 * float(iteration) / max(float(max_iterations), 1.0) - 1.0,
            dtype=torch.float32,
            device=device,
        )
        normalized_timestep = torch.tensor(
            2.0 * float(diffusion_timestep) / max(float(max_diffusion_timestep), 1.0) - 1.0,
            dtype=torch.float32,
            device=device,
        )
        previous_action_feature = self._normalize_action(previous_action, action_count, device)
        previous_consistency_feature = self._bounded_log_feature(
            torch.tensor(previous_consistency, dtype=torch.float32, device=device)
        )

        features = torch.stack(
            [
                self._bounded_log_feature(fidelity),
                self._bounded_log_feature(grad_norm),
                self._bounded_log_feature(measurement_energy),
                self._bounded_log_feature(x_energy),
                self._bounded_log_feature(spatial_variance),
                self._bounded_log_feature(latent_energy),
                torch.tanh(convergence_ratio),
                torch.clamp(cosine_alignment, -1.0, 1.0),
                self._bounded_log_feature(residual_to_signal),
                torch.clamp(normalized_iteration, -1.0, 1.0),
                torch.clamp(normalized_timestep, -1.0, 1.0),
                torch.clamp(previous_action_feature, -1.0, 1.0),
                torch.clamp(previous_consistency_feature, -1.0, 1.0),
            ]
        ).float()

        return torch.clamp(features, -self.state_clip, self.state_clip)
