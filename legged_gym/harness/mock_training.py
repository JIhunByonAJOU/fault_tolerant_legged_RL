"""Tiny subprocess used to smoke-test the harness without Isaac Gym."""

import argparse
import json
import signal
import time
from pathlib import Path


def _append(path, value):
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, sort_keys=True) + "\n")
        handle.flush()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--mode", choices=("normal", "error", "trend"), required=True)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--delay", type=float, default=0.05)
    args = parser.parse_args()
    metrics_path = Path(args.run_dir) / "metrics.jsonl"
    stopped = {"value": False}

    def request_stop(_signum, _frame):
        stopped["value"] = True

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)

    steps = args.steps if args.mode != "trend" else max(args.steps, 200)
    for iteration in range(steps):
        if stopped["value"]:
            return 130
        if args.mode == "normal":
            fraction = iteration / max(steps - 1, 1)
            vx_rmse = 0.50 - 0.35 * fraction
            vertical = 0.45 - 0.25 * fraction
            four_feet = 0.75 - 0.55 * fraction
        else:
            vx_rmse = 0.48
            vertical = 0.44
            four_feet = 0.74
        _append(
            metrics_path,
            {
                "iteration": iteration,
                "total_transitions": (iteration + 1) * 1000,
                "Rollout/command_vx_rmse": vx_rmse,
                "Rollout/vertical_velocity_rmse": vertical,
                "Rollout/four_feet_contact_rate": four_feet,
                "PPO/max_kl": 0.01,
                "PPO/nonfinite_update_skipped": 0.0,
            },
        )
        print("mock iteration {} mode {}".format(iteration, args.mode), flush=True)
        if iteration == 3:
            checkpoint = Path(args.run_dir) / "model_4.pt"
            checkpoint.write_bytes(b"mock-checkpoint")
        if args.mode == "error" and iteration == 2:
            raise RuntimeError("intentional harness smoke failure")
        time.sleep(args.delay)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
