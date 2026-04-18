from __future__ import annotations

from collections import Counter

import torch

from agents.agent import PPOAgent
from environment.state_builder import StateBuilder
from utils.ppo_utils import build_solver_library, load_agent_checkpoint


class PPO:
    """Inference-time PPO solver selector over reverse-diffusion steps."""

    def __init__(
        self,
        model: torch.nn.Module,
        agent_weights: str,
        device: str | torch.device,
        img_size: int = 32,
        noise_steps: int = 1000,
        channels: int = 3,
        hidden_dim: int = 256,
        ddnm_eta: float = 1.0,
        dps_scale: float = 0.0125,
        cg_iters: int = 5,
        diffpir_lambda: float = 1.0,
        noise_level_img: float = 0.0,
        diffpir_eta: float = 0.0,
        diffpir_zeta: float = 1.0,
    ) -> None:
        self.device = torch.device(device)
        self.noise_steps = int(noise_steps)
        self.state_builder = StateBuilder()
        self.solver_library = build_solver_library(
            model=model,
            device=self.device,
            image_size=img_size,
            channels=channels,
            diffusion_steps=self.noise_steps,
            ddnm_eta=ddnm_eta,
            dps_scale=dps_scale,
            diffpir_cg_iters=cg_iters,
            diffpir_lambda=diffpir_lambda,
            diffpir_noise_level_img=noise_level_img,
            diffpir_eta=diffpir_eta,
            diffpir_zeta=diffpir_zeta,
        )
        self.agent = PPOAgent(
            state_dim=self.state_builder.state_dim,
            action_dim=self.solver_library.action_dim,
            hidden_dim=hidden_dim,
        ).to(self.device)
        load_agent_checkpoint(self.agent, agent_weights, self.device)
        self.agent.eval()

    @staticmethod
    def _consistency_mse(x_estimate: torch.Tensor, y: torch.Tensor, operator) -> float:
        residual = operator.forward_pass(x_estimate) - y
        return float(torch.mean(residual.square()).item())

    def sample(self, y: torch.Tensor, operator) -> tuple[torch.Tensor, dict]:
        y = y.to(self.device)
        x_t = torch.randn(
            (1, self.solver_library.ddnm_solver.channels, self.solver_library.ddnm_solver.img_size, self.solver_library.ddnm_solver.img_size),
            device=self.device,
        )
        x_estimate = torch.clamp(operator.transpose_pass(y).detach().to(self.device), -1.0, 1.0)
        previous_estimate = None
        previous_action = 0
        previous_consistency = self._consistency_mse(x_estimate, y, operator)

        actions: list[int] = []
        solvers: list[str] = []

        for iteration, timestep in enumerate(range(self.noise_steps - 1, -1, -1)):
            state = self.state_builder.build(
                y=y,
                H=operator,
                x_latent=x_t,
                x_estimate=x_estimate,
                previous_estimate=previous_estimate,
                iteration=iteration,
                max_iterations=self.noise_steps,
                diffusion_timestep=timestep,
                max_diffusion_timestep=max(self.noise_steps - 1, 1),
                previous_action=previous_action,
                action_count=self.solver_library.action_dim,
                previous_consistency=previous_consistency,
            )

            with torch.no_grad():
                policy_step = self.agent.select_action(state.unsqueeze(0), deterministic=True)

            result = self.solver_library.apply_solver_step(
                action=policy_step.action,
                x_t=x_t,
                timestep=timestep,
                y=y,
                Phi=operator,
            )

            previous_estimate = x_estimate
            x_t = result.x_prev.detach()
            x_estimate = torch.clamp(result.x0_estimate.detach(), -1.0, 1.0)
            previous_action = policy_step.action
            previous_consistency = self._consistency_mse(x_estimate, y, operator)

            solver_name = self.solver_library.get_solver(policy_step.action).name
            actions.append(policy_step.action)
            solvers.append(solver_name)

        counts = Counter(solvers)
        return x_estimate, {"actions": actions, "solvers": solvers, "counts": dict(counts)}
