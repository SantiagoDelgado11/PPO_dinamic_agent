from __future__ import annotations

from dataclasses import dataclass

from solvers.base_diffusion_solver import DiffusionStepResult


@dataclass
class SolverLibrary:
    """Discrete solver action space for timestep-level control."""

    diffpir_solver: object
    dps_solver: object
    ddnm_solver: object

    def __post_init__(self) -> None:
        self._action_to_solver = {
            0: self.ddnm_solver,
            1: self.dps_solver,
            2: self.diffpir_solver,
        }
        for action, solver in self._action_to_solver.items():
            if not callable(getattr(solver, "step", None)):
                raise TypeError(
                    f"Solver mapped to action {action} ({type(solver).__name__}) "
                    "must implement a callable `step` method."
                )

    def _validate_action(self, action: int) -> None:
        if action not in self._action_to_solver:
            raise ValueError(f"Invalid action {action}. Expected one of {list(self._action_to_solver)}")

    def apply_solver_step(
        self,
        action: int,
        x_t,
        timestep: int,
        y,
        Phi,
    ) -> DiffusionStepResult:
        self._validate_action(action)
        solver = self._action_to_solver[action]
        return solver.step(x_t=x_t, timestep=timestep, y=y, H=Phi)

    def get_solver(self, action: int):
        self._validate_action(action)
        return self._action_to_solver[action]

    @property
    def action_dim(self) -> int:
        return len(self._action_to_solver)

    @property
    def solver_names(self) -> list[str]:
        return [self._action_to_solver[idx].name for idx in range(self.action_dim)]
