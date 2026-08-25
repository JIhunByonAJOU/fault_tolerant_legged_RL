"""Pure evaluation helpers for the precommitted P1 normal-gait gate."""

import hashlib
import json
import os
import uuid
from pathlib import Path

import numpy as np

try:
    import torch
except ImportError:  # CPU protocol tests intentionally remain Isaac/Torch independent.
    torch = None


SCENARIO_COMMANDS = {
    "stand": (0.0, 0.0, 0.0),
    "forward_slow": (0.25, 0.0, 0.0),
    "forward_nominal": (0.5, 0.0, 0.0),
    "lateral_left": (0.0, 0.2, 0.0),
    "lateral_right": (0.0, -0.2, 0.0),
    "arc_left": (0.3, 0.0, 0.3),
    "arc_right": (0.3, 0.0, -0.3),
    "heldout_randomized": (0.0, 0.0, 0.0),
}
ZERO_YAW_MOVING = {
    "forward_slow",
    "forward_nominal",
    "lateral_left",
    "lateral_right",
}
ARC_SCENARIOS = {"arc_left", "arc_right"}
HELDOUT_SCENARIO = "heldout_randomized"
HELDOUT_SEED_NAMESPACE = "p1-normal-gait-heldout-v1"


def heldout_command_sequence(seed, num_envs, generator):
    """Return deterministic eval-only commands from a disjoint seed namespace."""
    required = {
        "namespace",
        "num_segments",
        "segment_seconds",
        "lin_vel_x_range",
        "lin_vel_y_range",
        "yaw_rate_range",
        "trajectory_rmse_normalization_floor_m",
    }
    missing = required.difference(generator)
    if missing:
        raise ValueError("heldout generator missing keys: {}".format(sorted(missing)))
    if generator["namespace"] != HELDOUT_SEED_NAMESPACE:
        raise ValueError("unexpected heldout seed namespace")
    if generator["num_segments"] < 2 or generator["segment_seconds"] <= 0:
        raise ValueError("heldout piecewise protocol must contain positive segments")
    ranges = (
        ("lin_vel_x_range", 0.0, 0.8),
        ("lin_vel_y_range", -0.3, 0.3),
        ("yaw_rate_range", -0.5, 0.5),
    )
    for name, lower, upper in ranges:
        values = generator[name]
        if len(values) != 2 or not lower <= values[0] <= values[1] <= upper:
            raise ValueError("heldout {} exceeds training envelope".format(name))
    num_segments = int(generator["num_segments"])
    sequence = np.empty((int(num_segments), int(num_envs), 3), dtype=np.float32)
    for env_id in range(int(num_envs)):
        identity = "{}:{}:{}".format(generator["namespace"], int(seed), env_id)
        namespace_seed = int(hashlib.sha256(identity.encode("utf-8")).hexdigest()[:8], 16)
        rng = np.random.RandomState(namespace_seed)
        sequence[:, env_id, 0] = rng.uniform(*generator["lin_vel_x_range"], size=num_segments)
        sequence[:, env_id, 1] = rng.uniform(*generator["lin_vel_y_range"], size=num_segments)
        sequence[:, env_id, 2] = rng.uniform(*generator["yaw_rate_range"], size=num_segments)
    return sequence


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic_write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp-" + uuid.uuid4().hex)
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(str(temporary), str(path))


def quantile_summary(values):
    if torch is not None and isinstance(values, torch.Tensor):
        values = values.detach().cpu().numpy()
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if values.size == 0 or not np.isfinite(values).all():
        raise ValueError("quantile input must be non-empty and finite")
    return {
        "median": float(np.quantile(values, 0.5)),
        "p10": float(np.quantile(values, 0.1)),
        "p90": float(np.quantile(values, 0.9)),
    }


def normalized_trajectory_rmse(actual_xy, target_xy, target_path_length, floor_m=1.0):
    actual_xy = np.asarray(actual_xy, dtype=np.float64)
    target_xy = np.asarray(target_xy, dtype=np.float64)
    if actual_xy.shape != target_xy.shape or actual_xy.ndim != 3 or actual_xy.shape[-1] != 2:
        raise ValueError("trajectory arrays must have matching [steps, envs, 2] shape")
    rmse = np.sqrt(np.mean(np.sum(np.square(actual_xy - target_xy), axis=-1), axis=0))
    return rmse / np.maximum(np.asarray(target_path_length, dtype=np.float64), float(floor_m))


class ContactCycleAccumulator:
    """Accumulate one-step OR-filtered 1 N contacts for completed valid swings."""

    def __init__(
        self,
        num_envs,
        num_feet,
        dt,
        device="cpu",
        valid_swing_duration=(0.08, 1.50),
        contact_force_threshold=1.0,
        env_foot_duty_range=(0.30, 0.90),
        environment_valid_swing_count_min=2,
    ):
        self.num_envs = int(num_envs)
        self.num_feet = int(num_feet)
        self.dt = float(dt)
        self.valid_swing_duration = tuple(float(value) for value in valid_swing_duration)
        self.contact_force_threshold = float(contact_force_threshold)
        self.env_foot_duty_range = tuple(float(value) for value in env_foot_duty_range)
        self.environment_valid_swing_count_min = float(environment_valid_swing_count_min)
        shape = (self.num_envs, self.num_feet)
        self._torch_backend = torch is not None
        if self._torch_backend:
            self.previous_raw = torch.zeros(shape, dtype=torch.bool, device=device)
            self.previous_filtered = torch.zeros(shape, dtype=torch.bool, device=device)
            self.contact_steps = torch.zeros(shape, dtype=torch.long, device=device)
            self.measured_steps = torch.zeros(shape, dtype=torch.long, device=device)
            self.swing_steps = torch.zeros(shape, dtype=torch.long, device=device)
            self.valid_swings = torch.zeros(shape, dtype=torch.long, device=device)
        else:
            self.previous_raw = np.zeros(shape, dtype=bool)
            self.previous_filtered = np.zeros(shape, dtype=bool)
            self.contact_steps = np.zeros(shape, dtype=np.int64)
            self.measured_steps = np.zeros(shape, dtype=np.int64)
            self.swing_steps = np.zeros(shape, dtype=np.int64)
            self.valid_swings = np.zeros(shape, dtype=np.int64)

    def update(self, contact_forces_z, active=None):
        if self._torch_backend:
            raw = (
                torch.as_tensor(contact_forces_z, device=self.previous_raw.device)
                > self.contact_force_threshold
            )
        else:
            raw = np.asarray(contact_forces_z) > self.contact_force_threshold
        if raw.shape != self.previous_raw.shape:
            raise ValueError("contact shape mismatch")
        filtered = raw | self.previous_raw
        if active is None:
            active = (
                torch.ones(self.num_envs, dtype=torch.bool, device=raw.device)
                if self._torch_backend
                else np.ones(self.num_envs, dtype=bool)
            )
        if self._torch_backend:
            active = torch.as_tensor(active, dtype=torch.bool, device=raw.device).view(-1, 1)
            active = active.expand_as(raw)
            self.measured_steps += active.long()
            self.contact_steps += (filtered & active).long()
        else:
            active = np.broadcast_to(np.asarray(active, dtype=bool).reshape(-1, 1), raw.shape)
            self.measured_steps += active.astype(np.int64)
            self.contact_steps += (filtered & active).astype(np.int64)

        started_swing = self.previous_filtered & ~filtered & active
        in_swing = ~filtered & active
        self.swing_steps[started_swing] = 1
        continuing = in_swing & ~started_swing
        self.swing_steps[continuing] += 1
        ended = ~self.previous_filtered & filtered & active
        duration = (
            self.swing_steps.float() if self._torch_backend else self.swing_steps.astype(float)
        ) * self.dt
        valid = ended & (duration >= self.valid_swing_duration[0]) & (
            duration <= self.valid_swing_duration[1]
        )
        self.valid_swings += valid.long() if self._torch_backend else valid.astype(np.int64)
        self.swing_steps[ended] = 0
        if self._torch_backend:
            self.previous_raw.copy_(raw)
            self.previous_filtered.copy_(filtered)
        else:
            self.previous_raw[...] = raw
            self.previous_filtered[...] = filtered

    def result(self):
        if self._torch_backend:
            duty = self.contact_steps.float() / self.measured_steps.clamp_min(1).float()
            swings = self.valid_swings.float()
        else:
            duty = self.contact_steps.astype(float) / np.maximum(self.measured_steps, 1)
            swings = self.valid_swings.astype(float)
        per_foot_duty = [quantile_summary(duty[:, foot]) for foot in range(self.num_feet)]
        per_foot_swings = [quantile_summary(swings[:, foot]) for foot in range(self.num_feet)]
        duty_pair_ok = (duty >= self.env_foot_duty_range[0]) & (
            duty <= self.env_foot_duty_range[1]
        )
        swing_pair_ok = swings >= self.environment_valid_swing_count_min
        env_all = (
            duty_pair_ok.all(dim=1) & swing_pair_ok.all(dim=1)
            if self._torch_backend
            else duty_pair_ok.all(axis=1) & swing_pair_ok.all(axis=1)
        )
        return {
            "duty_factor_per_foot": per_foot_duty,
            "valid_swing_count_per_foot": per_foot_swings,
            "env_foot_duty_in_0p30_0p90_rate": (
                duty_pair_ok.float().mean().item()
                if self._torch_backend
                else float(duty_pair_ok.mean())
            ),
            "four_foot_cycle_pass_rate": (
                env_all.float().mean().item()
                if self._torch_backend
                else float(env_all.mean())
            ),
        }


def load_gate(path):
    path = Path(path)
    gate = json.loads(path.read_text(encoding="utf-8"))
    if gate.get("scenarios") != list(SCENARIO_COMMANDS):
        raise ValueError("gate scenarios do not match the precommitted order")
    if gate.get("seeds") != [1, 2, 3] or gate.get("num_envs") != 512:
        raise ValueError("gate seed/environment protocol mismatch")
    return gate


def expected_cell_keys(gate):
    return {
        (scenario, int(seed))
        for scenario in gate["scenarios"]
        for seed in gate["seeds"]
    }


def verify_matrix_cells(cells, gate, gate_sha256):
    keyed = {}
    errors = []
    for cell in cells:
        key = (cell.get("scenario"), int(cell.get("seed", -1)))
        if key in keyed:
            errors.append("duplicate cell {} seed {}".format(*key))
        keyed[key] = cell
        if cell.get("gate_sha256") != gate_sha256:
            errors.append("gate hash mismatch for {} seed {}".format(*key))
        if cell.get("num_envs") != gate["num_envs"]:
            errors.append("num_envs mismatch for {} seed {}".format(*key))
        if not cell.get("finite", False) or cell.get("return_code", 0) != 0:
            errors.append("nonfinite or nonzero cell {} seed {}".format(*key))
    expected = expected_cell_keys(gate)
    missing = expected.difference(keyed)
    extra = set(keyed).difference(expected)
    if missing:
        errors.append("missing cells: {}".format(sorted(missing)))
    if extra:
        errors.append("unexpected cells: {}".format(sorted(extra)))
    metadata = {
        json.dumps(
            {
                "protocol": cell.get("protocol"),
                "gate_sha256": cell.get("gate_sha256"),
                "checkpoint": cell.get("checkpoint"),
                "source_preflight": cell.get("source_preflight"),
                "num_envs": cell.get("num_envs"),
            },
            sort_keys=True,
        )
        for cell in cells
    }
    if len(metadata) != 1:
        errors.append("cell protocol metadata is not identical")
    return {"valid": not errors, "errors": errors, "cell_count": len(keyed)}


def gate_cell(cell, gate):
    """Apply every hard clause to one already-aggregated scenario/seed cell."""
    scenario = cell["scenario"]
    limits = gate["thresholds"]
    failures = []

    def maximum(metric, median, p90):
        value = cell["metrics"][metric]
        if value["median"] > median or value["p90"] > p90:
            failures.append(metric)

    def minimum(metric, median, p10):
        value = cell["metrics"][metric]
        if value["median"] < median or value["p10"] < p10:
            failures.append(metric)

    if cell["survival_rate"] < limits["survival_rate_min"]:
        failures.append("survival_rate")
    if scenario == "stand":
        for metric, bounds in limits["stand_max"].items():
            maximum(metric, bounds["median"], bounds["p90"])
    else:
        for metric, bounds in limits["moving_max"].items():
            maximum(metric, bounds["median"], bounds["p90"])
        if scenario in ZERO_YAW_MOVING:
            for metric, bounds in limits["zero_yaw_max"].items():
                maximum(metric, bounds["median"], bounds["p90"])
            for metric, bounds in limits["zero_yaw_min"].items():
                minimum(metric, bounds["median"], bounds["p10"])
        if scenario in ARC_SCENARIOS:
            arc = limits["arc"]
            if cell["correct_yaw_sign_rate"] < arc["correct_yaw_sign_rate_min"]:
                failures.append("correct_yaw_sign_rate")
            bounds = arc["abs_integrated_yaw_error_rad_max"]
            maximum("abs_integrated_yaw_error_rad", bounds["median"], bounds["p90"])
        if scenario == HELDOUT_SCENARIO:
            bounds = limits["heldout_randomized_max"]["normalized_trajectory_rmse"]
            maximum("normalized_trajectory_rmse", bounds["median"], bounds["p90"])
        gait = cell["gait_cycles"]
        gait_limits = limits["gait_cycle"]
        if any(
            not (
                gait_limits["duty_factor_population_median_min"]
                <= foot["median"]
                <= gait_limits["duty_factor_population_median_max"]
            )
            for foot in gait["duty_factor_per_foot"]
        ):
            failures.append("median_duty_factor")
        if (
            gait["env_foot_duty_in_0p30_0p90_rate"]
            < gait_limits["env_foot_duty_in_0p30_0p90_rate_min"]
        ):
            failures.append("env_foot_duty_rate")
        if any(
            foot["median"] < gait_limits["valid_swing_count_median_min"]
            or foot["p10"] < gait_limits["valid_swing_count_p10_min"]
            for foot in gait["valid_swing_count_per_foot"]
        ):
            failures.append("valid_swing_count")
        if gait["four_foot_cycle_pass_rate"] < gait_limits["four_foot_cycle_pass_rate_min"]:
            failures.append("four_foot_cycle_pass_rate")
    return {"passed": not failures, "failures": sorted(set(failures))}
