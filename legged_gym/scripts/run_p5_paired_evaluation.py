"""Run the frozen P5 Teacher/Student paired evaluation protocol."""

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PROTOCOL = ROOT / "legged_gym" / "evaluation" / "p5_paired_protocol.json"
DEFAULT_OUTPUT = ROOT / "logs" / "evaluations" / "p5-paired-tf43000-jt-students-v1"


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_protocol(path):
    with path.open(encoding="utf-8") as stream:
        protocol = json.load(stream)
    if protocol.get("schema_version") != 1:
        raise ValueError("unsupported P5 protocol schema")
    model_ids = [model["id"] for model in protocol.get("models", [])]
    if not model_ids or len(model_ids) != len(set(model_ids)):
        raise ValueError("P5 protocol model ids must be non-empty and unique")
    seeds = protocol.get("seeds", [])
    if not seeds or len(seeds) != len(set(seeds)):
        raise ValueError("P5 protocol seeds must be non-empty and unique")
    return protocol


def resolve_models(protocol, selected_ids=None):
    selected = set(selected_ids or [])
    known = {model["id"] for model in protocol["models"]}
    unknown = selected - known
    if unknown:
        raise ValueError("unknown model ids: {}".format(", ".join(sorted(unknown))))
    models = [model for model in protocol["models"] if not selected or model["id"] in selected]
    resolved = []
    for model in models:
        item = dict(model)
        checkpoint = (ROOT / item["checkpoint"]).resolve()
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        actual_sha = sha256_file(checkpoint)
        if actual_sha != item["checkpoint_sha256"]:
            raise ValueError(
                "checkpoint SHA mismatch for {}: {} != {}".format(
                    item["id"], actual_sha, item["checkpoint_sha256"]
                )
            )
        item["checkpoint_path"] = str(checkpoint)
        resolved.append(item)
    return resolved


def build_jobs(protocol, models, phases, smoke=False):
    jobs = []
    seeds = protocol["seeds"][:1] if smoke else protocol["seeds"]
    for model in models:
        for phase in phases:
            config = protocol[phase]
            for seed in seeds:
                name = "{}_seed{}_{}".format(model["id"], seed, phase)
                if smoke:
                    name += "_smoke"
                jobs.append(
                    {
                        "name": name,
                        "phase": phase,
                        "seed": int(seed),
                        "model": model,
                        "config": config,
                        "command_x": float(protocol["command_x_mps"]),
                        "smoke": smoke,
                    }
                )
    return jobs


def command_for_job(job, output):
    model = job["model"]
    config = job["config"]
    common = [
        sys.executable,
        "-m",
        "legged_gym.scripts.evaluate_teacher243_failure_{}".format(
            "matrix" if job["phase"] == "fixed" else "onset"
        ),
        "--checkpoint-path",
        model["checkpoint_path"],
        "--policy-mode",
        model["policy_mode"],
        "--command-x",
        str(job["command_x"]),
        "--output",
        str(output),
        "--task",
        model["task"],
        "--seed",
        str(job["seed"]),
        "--headless",
    ]
    if job["phase"] == "fixed":
        rates = [0.0, 1.0] if job["smoke"] else config["degradation_rates"]
        return common + [
            "--steps",
            str(10 if job["smoke"] else config["steps"]),
            "--replicates-per-condition",
            str(1 if job["smoke"] else config["replicates_per_condition"]),
            "--rates",
            *[str(value) for value in rates],
        ]
    rates = [1.0] if job["smoke"] else config["degradation_rates"]
    onsets = [0.1] if job["smoke"] else config["onset_seconds"]
    return common + [
        "--onset-seconds",
        *[str(value) for value in onsets],
        "--post-seconds",
        str(0.1 if job["smoke"] else config["post_seconds"]),
        "--replicates-per-condition",
        str(1 if job["smoke"] else config["replicates_per_condition"]),
        "--rates",
        *[str(value) for value in rates],
        "--recovery-window-seconds",
        str(config["recovery_window_seconds"]),
        "--recovery-vx-error",
        str(config["recovery_vx_error_mps"]),
        "--recovery-yaw-rate",
        str(config["recovery_yaw_rate_radps"]),
    ]


def atomic_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(str(temporary), str(path))


def evaluation_environment():
    environment = os.environ.copy()
    # Isaac Gym still uses deprecated NumPy aliases.  The WIM environment pins
    # a compatible NumPy, while the user's site-packages may shadow it.
    environment["PYTHONNOUSERSITE"] = "1"
    return environment


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--models", nargs="+")
    parser.add_argument("--phases", nargs="+", choices=("fixed", "onset"), default=("fixed", "onset"))
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true", help="Skip existing non-empty result files.")
    return parser.parse_args()


def main():
    args = parse_args()
    protocol_path = args.protocol.resolve()
    output_dir = args.output_dir.resolve()
    protocol = load_protocol(protocol_path)
    models = resolve_models(protocol, args.models)
    jobs = build_jobs(protocol, models, args.phases, smoke=args.smoke)
    result_dir = output_dir / ("smoke" if args.smoke else "canonical")
    manifest = {
        "schema_version": 1,
        "protocol": str(protocol_path),
        "protocol_sha256": sha256_file(protocol_path),
        "output_dir": str(output_dir),
        "smoke": args.smoke,
        "jobs": [],
    }
    for job in jobs:
        output = result_dir / (job["name"] + ".jsonl")
        log = result_dir / (job["name"] + ".log")
        command = command_for_job(job, output)
        record = {
            "name": job["name"],
            "phase": job["phase"],
            "model": job["model"]["id"],
            "seed": job["seed"],
            "output": str(output),
            "log": str(log),
            "command": command,
            "status": "planned",
        }
        manifest["jobs"].append(record)
        if args.dry_run:
            print(" ".join(command))
            continue
        if args.resume and output.is_file() and output.stat().st_size > 0:
            record["status"] = "skipped_existing"
            continue
        result_dir.mkdir(parents=True, exist_ok=True)
        with log.open("w", encoding="utf-8") as stream:
            process = subprocess.run(
                command,
                cwd=ROOT,
                env=evaluation_environment(),
                text=True,
                stdout=stream,
                stderr=subprocess.STDOUT,
            )
        record["returncode"] = process.returncode
        record["status"] = "complete" if process.returncode == 0 else "failed"
        atomic_json(output_dir / "manifest.json", manifest)
        if process.returncode != 0:
            raise RuntimeError("P5 job failed: {} (see {})".format(job["name"], log))
    atomic_json(output_dir / "manifest.json", manifest)


if __name__ == "__main__":
    main()
