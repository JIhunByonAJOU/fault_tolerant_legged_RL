"""Fail-closed source, configuration, and tensor conformance checks."""

import hashlib
import ast
import inspect
import copy
import importlib.util
import tempfile
import json
import os
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
OFFICIAL_LEGGED_GYM_COMMIT = "ae614c029977157123225f538ecdd3f873e54bd4"
RSL_RL_ROOT = Path("/home/jihun/Capstone2/rsl_rl")
OFFICIAL_RSL_RL_COMMIT = "2ad79cf0caa85b91721abfe358105f869a784121"
RSL_RL_HASHES = {
    "rsl_rl/modules/actor_critic.py": "7113dd5cfcc8a7220400b50e07a33fd7261cea31c781728156051ea36af0b991",
    "rsl_rl/algorithms/ppo.py": "33d531858767e49bbffce8030ac2879a30a00a64caeef0f87502285fa15301cd",
    "rsl_rl/runners/on_policy_runner.py": "3a704d0cdf74c4637dc34e25660de122308a0443a0a91eff03b1bbcafe3fb8d7",
}
ALLOWED_COMPATIBILITY_PATHS = {
    "legged_gym/envs/a1/a1_config.py",
    "legged_gym/envs/base/legged_robot.py",
    "legged_gym/envs/base/legged_robot_config.py",
}
TERRAIN_PATH = "legged_gym/utils/terrain.py"
_TERRAIN_RECEIVER_SUBSTITUTIONS = (
    ("vertical_scale=self.vertical_scale", "vertical_scale=self.cfg.vertical_scale"),
    ("horizontal_scale=self.horizontal_scale", "horizontal_scale=self.cfg.horizontal_scale"),
)


def _run(argv, cwd):
    process = subprocess.run(argv, cwd=str(cwd), text=True, encoding="utf-8", stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if process.returncode:
        raise RuntimeError("command failed: {}: {}".format(" ".join(argv), process.stderr.strip()))
    return process.stdout


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def exact_terrain_receiver_compatibility(official_source, current_source):
    """Accept only the pinned terrain source with both selected-terrain receiver fixes."""
    try:
        tree = ast.parse(official_source)
        terrain_class = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "Terrain")
        selected = next(node for node in terrain_class.body if isinstance(node, ast.FunctionDef) and node.name == "selected_terrain")
        selected_source = ast.get_source_segment(official_source, selected)
    except (SyntaxError, StopIteration):
        return False
    if selected_source is None:
        return False
    accepted = official_source
    for original, corrected in _TERRAIN_RECEIVER_SUBSTITUTIONS:
        if selected_source.count(original) != 1 or official_source.count(original) != 1:
            return False
        if corrected in selected_source:
            return False
        accepted = accepted.replace(original, corrected)
    return current_source == accepted


def compare_legged_gym_revision(official_commit=OFFICIAL_LEGGED_GYM_COMMIT):
    _run(["git", "cat-file", "-e", official_commit + "^{commit}"], REPO_ROOT)
    scientific_paths = [
        "legged_gym/envs/a1/a1_config.py",
        "legged_gym/envs/base/legged_robot.py",
        "legged_gym/envs/base/legged_robot_config.py",
        TERRAIN_PATH,
    ]
    names = _run(["git", "diff", "--name-only", official_commit, "--"] + scientific_paths, REPO_ROOT).splitlines()
    forbidden = sorted(set(names) - ALLOWED_COMPATIBILITY_PATHS)
    allowed = set(names) & ALLOWED_COMPATIBILITY_PATHS
    terrain_compatibility = False
    if TERRAIN_PATH in names:
        official_terrain = _run(["git", "show", official_commit + ":" + TERRAIN_PATH], REPO_ROOT)
        current_terrain = (REPO_ROOT / TERRAIN_PATH).read_text(encoding="utf-8")
        terrain_compatibility = exact_terrain_receiver_compatibility(official_terrain, current_terrain)
        if terrain_compatibility:
            forbidden.remove(TERRAIN_PATH)
            allowed.add(TERRAIN_PATH)
    audited_paths = [path for path in scientific_paths if path != TERRAIN_PATH or not terrain_compatibility]
    diff = _run(["git", "diff", official_commit, "--"] + audited_paths, REPO_ROOT)
    required_compatibility_markers = (
        '+        name = "a1"',
        "+        name = \"legged_robot\"",
        "+            actor_handle = self.gym.create_actor",
    )
    unexpected = []
    for line in diff.splitlines():
        if not line.startswith(("+", "-")) or line.startswith(("+++", "---")):
            continue
        if any(marker in line for marker in required_compatibility_markers):
            continue
        compatibility_fragments = (
            "HeightFieldProperties", "HeightFieldParams", "self.terrain.horizontal_scale",
            "self.terrain.vertical_scale", "self.terrain.border_size", "self.terrain.cfg.",
            "anymal_handle", "actor_handle", "create_actor", "No newline at end of file",
            "_reward_feet_contact_forces", "self.cfg.rewards.max_contact_force",
        )
        if not any(fragment in line for fragment in compatibility_fragments):
            unexpected.append(line)
    forbidden.extend(unexpected)
    return {
        "official_revision": official_commit,
        "current_head": _run(["git", "rev-parse", "HEAD"], REPO_ROOT).strip(),
        "allowed_diff_paths": sorted(allowed),
        "forbidden_diffs": forbidden,
        "pass": not forbidden,
    }


def verify_rsl_rl_revision(root=RSL_RL_ROOT, official_commit=OFFICIAL_RSL_RL_COMMIT):
    root = Path(root).resolve()
    revision = _run(["git", "rev-parse", "HEAD"], root).strip()
    status = _run(["git", "status", "--porcelain"], root)
    hashes = {name: _sha(root / name) for name in RSL_RL_HASHES}
    failures = []
    if revision != official_commit:
        failures.append("revision")
    if status:
        failures.append("dirty_worktree")
    if hashes != RSL_RL_HASHES:
        failures.append("semantic_source_hash")
    return {"root": str(root), "revision": revision, "clean": not status, "hashes": hashes, "failures": failures, "pass": not failures}


def run_actor_ppo_probe(device="cpu"):
    import torch
    from rsl_rl.algorithms import PPO
    from rsl_rl.modules import ActorCritic

    torch.manual_seed(1)
    actor = ActorCritic(235, 235, 12, actor_hidden_dims=[32], critic_hidden_dims=[32]).to(device)
    observations = torch.linspace(-1.0, 1.0, 4 * 235, device=device).reshape(4, 235)
    torch.manual_seed(7)
    sample = actor.act(observations)
    log_probability = actor.get_actions_log_prob(sample)
    inference = actor.act_inference(observations)
    passed = bool(torch.isfinite(sample).all() and torch.isfinite(log_probability).all() and torch.isfinite(inference).all())
    passed = passed and bool(torch.equal(inference, actor.action_mean))
    def one_update(seed):
        torch.manual_seed(seed)
        model = ActorCritic(235, 235, 12, actor_hidden_dims=[32], critic_hidden_dims=[32]).to(device)
        algorithm = PPO(model, num_learning_epochs=1, num_mini_batches=1, learning_rate=1e-3, schedule="fixed", device=device)
        algorithm.init_storage(2, 2, [235], [235], [12])
        for step in range(2):
            obs = torch.linspace(-0.5 + step, 0.5 + step, 2 * 235, device=device).reshape(2, 235)
            algorithm.act(obs, obs)
            algorithm.process_env_step(torch.tensor([0.25, -0.1], device=device), torch.zeros(2, device=device), {})
        algorithm.compute_returns(obs)
        before = [value.detach().clone() for value in model.state_dict().values()]
        losses = algorithm.update()
        after = list(model.state_dict().values())
        changed = any(not torch.equal(left, right) for left, right in zip(before, after))
        finite_losses = all(math.isfinite(float(value)) for value in losses)
        return losses, changed, finite_losses, len(algorithm.optimizer.state)
    import math
    first_update = one_update(11)
    second_update = one_update(11)
    update_deterministic = all(abs(float(a) - float(b)) < 1e-7 for a, b in zip(first_update[0], second_update[0]))
    update_pass = first_update[1] and first_update[2] and first_update[3] > 0 and update_deterministic
    passed = passed and update_pass
    return {
        "device": device,
        "sample_shape": list(sample.shape),
        "log_probability_shape": list(log_probability.shape),
        "inference_is_distribution_mean": bool(torch.equal(inference, actor.action_mean)),
        "finite": passed,
        "ppo_update": {"losses": list(first_update[0]), "parameters_changed": first_update[1], "optimizer_state_entries": first_update[3], "deterministic_repeat": update_deterministic, "pass": update_pass},
        "pass": passed,
    }


def verify_logging_only_adapter():
    path = REPO_ROOT / "legged_gym/learning/official_wim_runner.py"
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "OfficialWimOnPolicyRunner")
    bases = [base.id if isinstance(base, ast.Name) else ast.get_source_segment(source, base) for base in cls.bases]
    overrides = sorted(node.name for node in cls.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and not node.name.startswith("_"))
    log_node = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "log")
    log_source = ast.get_source_segment(source, log_node)
    super_index = log_source.index("super().log")
    append_index = log_source.index("self._append_metric_row")
    passed = bases == ["OnPolicyRunner"] and overrides == ["log"] and super_index < append_index
    state_equivalent = False
    try:
        import torch
        from types import SimpleNamespace
        from unittest import mock
        from rsl_rl.runners import OnPolicyRunner
        spec = importlib.util.spec_from_file_location("_official_wim_runner_probe", str(path))
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        runner = object.__new__(module.OfficialWimOnPolicyRunner)
        parameter = torch.nn.Parameter(torch.tensor([1.0]))
        optimizer = torch.optim.Adam([parameter], lr=1e-3)
        actor = SimpleNamespace(std=torch.ones(12), state_dict=lambda: {"parameter": parameter.detach().clone()})
        runner.alg = SimpleNamespace(actor_critic=actor, optimizer=optimizer, storage=SimpleNamespace(observations=torch.ones(1)), learning_rate=1e-3)
        runner.num_steps_per_env = 24
        runner.env = SimpleNamespace(num_envs=2)
        runner.tot_timesteps = 0
        runner.log_dir = tempfile.mkdtemp(prefix="official-wim-adapter-probe-")
        runner._wandb_run = SimpleNamespace(log=lambda row, step: None)
        before = (copy.deepcopy(actor.state_dict()), copy.deepcopy(optimizer.state_dict()), runner.alg.storage.observations.clone())
        locs = {"it": 0, "collection_time": 0.1, "learn_time": 0.2, "mean_value_loss": 1.0, "mean_surrogate_loss": 0.5, "rewbuffer": [], "lenbuffer": [], "ep_infos": []}
        with mock.patch.object(OnPolicyRunner, "log", return_value=None):
            runner.log(locs)
        after = (actor.state_dict(), optimizer.state_dict(), runner.alg.storage.observations)
        state_equivalent = torch.equal(before[0]["parameter"], after[0]["parameter"]) and before[1] == after[1] and torch.equal(before[2], after[2])
    except Exception:
        state_equivalent = False
    passed = passed and state_equivalent
    return {"base_classes": bases, "public_overrides": overrides, "super_log_first": super_index < append_index, "model_optimizer_storage_state_equivalent": state_equivalent, "source_sha256": _sha(path), "pass": passed}


def pre_environment_conformance():
    legged = compare_legged_gym_revision()
    rsl = verify_rsl_rl_revision()
    if not legged["pass"] or not rsl["pass"]:
        raise RuntimeError("official source conformance failed: {}".format({"legged_gym": legged, "rsl_rl": rsl}))
    return {"legged_gym": legged, "rsl_rl": rsl}


def build_provenance(args, conformance):
    status = _run(["git", "status", "--porcelain=v1"], REPO_ROOT)
    diff = _run(["git", "diff", "--binary", "HEAD"], REPO_ROOT)
    adapter = REPO_ROOT / "legged_gym/learning/official_wim_runner.py"
    return {
        "task": "a1_official_wim_rough",
        "cli": list(sys.argv),
        "seed": args.seed,
        "devices": {"sim": args.sim_device, "rl": args.rl_device},
        "budget": {"num_envs": args.num_envs, "max_iterations": args.max_iterations, "num_steps_per_env": args.num_steps_per_env},
        "resume": False,
        "no_resume": {"resume": False, "load_run": None, "checkpoint": None},
        "repository": {"head": _run(["git", "rev-parse", "HEAD"], REPO_ROOT).strip(), "status": status.splitlines(), "diff_sha256": hashlib.sha256(diff.encode()).hexdigest()},
        "official_sources": conformance,
        "adapter_sha256": _sha(adapter),
        "provenance_labels": {
            "Paper Explicit": ["five terrains", "8 m traversal", "episode-constant commands", "forward 0.75 m/s", "lateral [-0.1, 0.1] m/s", "cross without base contact"],
            "Implementation Choice - Agent recommendation": ["deterministic seeds", "Wilson eligibility", "selection tie-break"],
        },
    }


def complete_conformance(device="cpu"):
    results = {
        "legged_gym": compare_legged_gym_revision(),
        "rsl_rl": verify_rsl_rl_revision(),
        "actor_ppo_probe": run_actor_ppo_probe(device),
        "logging_adapter": verify_logging_only_adapter(),
    }
    results["pass"] = all(value.get("pass", False) for value in results.values())
    return results
