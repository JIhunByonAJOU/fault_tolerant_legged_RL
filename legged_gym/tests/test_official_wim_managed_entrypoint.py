import hashlib
import os
import runpy
import sys
import tempfile
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from legged_gym.scripts import train_official_wim_managed as managed


class FakeRun:
    def __init__(self, order):
        self.order = order
        self.config = self
    def update(self, value, allow_val_change=False):
        self.order.append("wandb.config.update")
    def finish(self):
        self.order.append("wandb.finish")


def args(path):
    return SimpleNamespace(task=managed.TASK, wandb=True, resume=False, load_run=None, checkpoint=None, log_dir=str(path), run_name=path.name, wandb_project="test", seed=1, sim_device="cuda:0", rl_device="cuda:0", num_envs=64, max_iterations=2, num_steps_per_env=24)


class ManagedEntrypointTest(unittest.TestCase):
    def test_direct_script_bootstrap_ignores_stale_sibling_checkout(self):
        repo_root = Path(__file__).resolve().parents[2]
        entrypoint = repo_root / "legged_gym/scripts/train_official_wim_managed.py"
        scripts_dir = entrypoint.parent
        stale_checkout = Path("/home/jihun/Capstone2/legged_gym")
        self.assertEqual(Path(managed.__file__).resolve(), entrypoint)
        self.assertEqual(Path(sys.modules["legged_gym"].__file__).resolve(), repo_root / "legged_gym/__init__.py")
        self.assertTrue((stale_checkout / "legged_gym/__init__.py").is_file())
        self.assertFalse((stale_checkout / "legged_gym/official_wim").exists())

        original_modules = {
            name: module
            for name, module in sys.modules.items()
            if name == "legged_gym" or name.startswith("legged_gym.")
        }
        original_path = list(sys.path)
        original_environment_path = os.environ.get("PATH")
        try:
            for name in original_modules:
                sys.modules.pop(name, None)
            sys.path[:] = [
                str(scripts_dir),
                str(stale_checkout),
                *[
                    entry
                    for entry in original_path
                    if Path(entry or os.curdir).resolve() not in {repo_root, stale_checkout.resolve()}
                ],
            ]
            namespace = runpy.run_path(str(entrypoint), run_name="managed_direct_script_probe")
            resolved_package = Path(sys.modules["legged_gym"].__file__).resolve()
            resolved_official_wim = Path(sys.modules["legged_gym.official_wim"].__file__).resolve()
            self.assertEqual(resolved_package, repo_root / "legged_gym/__init__.py")
            self.assertIn(repo_root, resolved_official_wim.parents)
            self.assertEqual(namespace["_REPO_ROOT"], repo_root)
        finally:
            for name in list(sys.modules):
                if name == "legged_gym" or name.startswith("legged_gym."):
                    sys.modules.pop(name, None)
            sys.modules.update(original_modules)
            sys.path[:] = original_path
            if original_environment_path is None:
                os.environ.pop("PATH", None)
            else:
                os.environ["PATH"] = original_environment_path

    def test_bootstrap_is_idempotent_for_repo_and_interpreter_bin(self):
        repo_root = Path(managed.__file__).resolve().parents[2]
        interpreter_bin = Path(sys.executable).resolve().parent
        with mock.patch.object(
            sys,
            "path",
            [str(repo_root), "/tmp/example", str(repo_root)],
        ), mock.patch.dict(
            os.environ,
            {"PATH": os.pathsep.join((str(interpreter_bin), "/usr/bin", str(interpreter_bin)))},
            clear=False,
        ):
            managed._bootstrap_managed_imports()
            managed._bootstrap_managed_imports()
            self.assertEqual(sys.path[0], str(repo_root))
            self.assertEqual(sys.path.count(str(repo_root)), 1)
            path_entries = os.environ["PATH"].split(os.pathsep)
            self.assertEqual(path_entries[0], str(interpreter_bin))
            self.assertEqual(path_entries.count(str(interpreter_bin)), 1)

    def test_wandb_failure_prevents_factories_and_artifacts(self):
        with tempfile.TemporaryDirectory() as root:
            run_dir = Path(root) / "run"; run_dir.mkdir()
            with mock.patch.object(managed, "run_pure_preflight", return_value={"log_dir": str(run_dir), "conformance": {}}), mock.patch.object(managed, "initialize_wandb_before_environment", side_effect=RuntimeError("offline")):
                with self.assertRaises(RuntimeError):
                    managed.train(args(run_dir))
            self.assertFalse((run_dir / "resolved_config.json").exists())
            self.assertFalse((run_dir / "metrics.jsonl").exists())
            self.assertEqual(list(run_dir.glob("model_*.pt")), [])

    def test_exact_order_and_learn_payload(self):
        order = []
        with tempfile.TemporaryDirectory() as root:
            run_dir = Path(root) / "run"; run_dir.mkdir()
            fake_run = FakeRun(order)
            env = SimpleNamespace(num_obs=235, num_privileged_obs=None, num_actions=12)
            runner = SimpleNamespace(_resolved_config={"ok": True}, learn=lambda **kw: order.append(("learn", kw)))
            registry = SimpleNamespace(
                make_env=lambda **kw: (order.append("make_env") or (env, SimpleNamespace())),
                make_alg_runner=lambda **kw: (order.append("make_alg_runner") or (runner, SimpleNamespace(runner=SimpleNamespace(max_iterations=2)))),
            )
            modules = {
                "legged_gym.envs": types.SimpleNamespace(task_registry=registry),
                "legged_gym.learning.official_wim_runner": types.SimpleNamespace(set_pre_environment_context=lambda value: order.append("context")),
                "legged_gym.utils.helpers": types.SimpleNamespace(class_to_dict=lambda value: {}),
            }
            with mock.patch.object(managed, "run_pure_preflight", return_value={"log_dir": str(run_dir), "conformance": {}}), mock.patch.object(managed, "initialize_wandb_before_environment", side_effect=lambda _: (order.append("wandb.init") or fake_run)), mock.patch.object(managed, "build_provenance", return_value={}), mock.patch.dict(sys.modules, modules):
                managed.train(args(run_dir))
            self.assertLess(order.index("wandb.init"), order.index("make_env"))
            self.assertLess(order.index("make_env"), order.index("make_alg_runner"))
            self.assertIn(("learn", {"num_learning_iterations": 2, "init_at_random_ep_len": True}), order)

    def test_mismatch_and_generic_entrypoint_guard(self):
        with tempfile.TemporaryDirectory() as root:
            value = args(Path(root) / "actual")
            value.run_name = "different"
            with self.assertRaises(ValueError):
                managed.run_pure_preflight(value)
        path = Path(__file__).resolve().parents[1] / "scripts/train.py"
        self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), "655d0ddb57c1576494d999126f241139422b3dff569f8d6c63ad83657684bc38")


if __name__ == "__main__":
    unittest.main()
