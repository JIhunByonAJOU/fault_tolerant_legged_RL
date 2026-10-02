"""Command-line interface shared by 슈퍼바이저, tracker, and smoke tests."""

import argparse
import json
import os
import sys
import tempfile
import time
from pathlib import Path

from .manager import (
    TERMINAL_STATES,
    HarnessError,
    RunState,
    RunStore,
    assess_trend,
    begin_verification,
    collect_run,
    finish_modification,
    gpu_snapshot,
    launch_run,
    mark_blocked,
    record_stage_error,
    reopen_analysis,
    resume_blocked_analysis,
    route_analysis,
    stop_run,
    supervise,
    validate_p0_gate,
)


def _print(value):
    print(json.dumps(value, indent=2, sort_keys=True))


def _wait_terminal(run_dir, timeout=10.0, trend_min_iteration=250):
    deadline = time.time() + timeout
    snapshot = None
    while time.time() < deadline:
        snapshot = collect_run(run_dir, stale_seconds=30, trend_min_iteration=trend_min_iteration)
        if snapshot["state"]["state"] in TERMINAL_STATES:
            return snapshot
        time.sleep(0.05)
    raise HarnessError("Timed out waiting for terminal run: {}".format(run_dir))


def _event_states(run_dir):
    states = []
    versions = []
    path = Path(run_dir) / "events.jsonl"
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            if "to" in record:
                states.append(record["to"])
                versions.append(record["version"])
    return states, versions


def _monitor_event(run_dir, snapshot, checkpoint_interval=0):
    """Persist and return only a decision-relevant monitor event."""
    run_dir = Path(run_dir)
    monitor_state_path = run_dir / "monitor_state.json"
    try:
        monitor_state = json.loads(monitor_state_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        monitor_state = {"sequence": 0, "last_fingerprint": None}

    state = snapshot["state"]
    hard_alerts = [
        item for item in snapshot.get("alerts", []) if item.get("severity") == "hard"
    ]
    checkpoints = snapshot.get("checkpoints", [])
    latest_checkpoint = checkpoints[-1] if checkpoints else None
    event_type = None
    if state["state"] in TERMINAL_STATES:
        event_type = "terminal"
    elif hard_alerts:
        event_type = "hard_alert"
    elif snapshot.get("trend", {}).get("candidate_stop"):
        event_type = "trend_candidate"
    elif checkpoint_interval and latest_checkpoint:
        iteration = int(latest_checkpoint["iteration"])
        last_notified = int(monitor_state.get("last_notified_checkpoint", 0))
        if iteration > 0 and iteration // checkpoint_interval > last_notified // checkpoint_interval:
            event_type = "checkpoint"

    fingerprint_value = {
        "type": event_type,
        "state": state["state"],
        "state_version": state.get("state_version"),
        "checkpoint": latest_checkpoint["iteration"] if latest_checkpoint else None,
        "hard_alerts": sorted(item.get("kind") for item in hard_alerts),
        "trend_iteration": snapshot.get("trend", {}).get("latest_iteration"),
    }
    fingerprint = json.dumps(fingerprint_value, sort_keys=True)
    event = None
    if event_type and fingerprint != monitor_state.get("last_fingerprint"):
        event = {
            "schema_version": 1,
            "sequence": int(monitor_state.get("sequence", 0)) + 1,
            "event_type": event_type,
            "recorded_at": snapshot["collected_at"],
            "run_id": snapshot["run_id"],
            "state": state["state"],
            "state_version": state.get("state_version"),
            "latest_iteration": snapshot.get("trend", {}).get("latest_iteration"),
            "latest_checkpoint": latest_checkpoint,
            "hard_alerts": hard_alerts,
            "trend": snapshot.get("trend"),
        }
        temporary = run_dir / ("monitor_event.json.tmp-{}".format(os.getpid()))
        temporary.write_text(json.dumps(event, indent=2, sort_keys=True), encoding="utf-8")
        os.replace(str(temporary), str(run_dir / "monitor_event.json"))
        with (run_dir / "monitor_events.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, sort_keys=True) + "\n")
            handle.flush()
        monitor_state["sequence"] = event["sequence"]
        monitor_state["last_fingerprint"] = fingerprint
        if event_type == "checkpoint" and latest_checkpoint:
            monitor_state["last_notified_checkpoint"] = latest_checkpoint["iteration"]

    monitor_state["last_collected_at"] = snapshot["collected_at"]
    temporary = run_dir / ("monitor_state.json.tmp-{}".format(os.getpid()))
    temporary.write_text(json.dumps(monitor_state, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(str(temporary), str(monitor_state_path))
    return event


def smoke_loop(output_root=None):
    """Exercise completion, crash, and trend-stop flows with no Isaac imports."""
    temporary = None
    if output_root is None:
        temporary = tempfile.TemporaryDirectory(prefix="legged-harness-smoke-")
        output_root = temporary.name
    repo_root = Path(__file__).resolve().parents[2]
    python = sys.executable
    results = {}
    try:
        for mode in ("normal", "error"):
            launched = launch_run(
                repo_root=repo_root,
                phase="p00-smoke",
                experiment=mode,
                label=mode,
                command=[
                    python,
                    "-m",
                    "legged_gym.harness.mock_training",
                    "--run-dir",
                    "{run_dir}",
                    "--mode",
                    mode,
                    "--steps",
                    "8",
                    "--delay",
                    "0.02",
                ],
                authority="supervisor",
                output_root=output_root,
                require_rtx4090=False,
                mock=True,
            )
            snapshot = _wait_terminal(launched["run_dir"])
            expected = RunState.COMPLETED.value if mode == "normal" else RunState.ERROR.value
            if snapshot["state"]["state"] != expected:
                raise HarnessError("{} smoke ended in {}".format(mode, snapshot["state"]["state"]))
            states, versions = _event_states(launched["run_dir"])
            if versions != sorted(set(versions)):
                raise HarnessError("Non-monotonic state versions in {} smoke".format(mode))
            results[mode] = {
                "state": expected,
                "run_id": launched["run_id"],
                "events": states,
                "analysis_requested": snapshot["state"]["analysis_requested"],
                "exit": snapshot["exit"],
            }

        launched = launch_run(
            repo_root=repo_root,
            phase="p00-smoke",
            experiment="trend",
            label="trend",
            command=[
                python,
                "-m",
                "legged_gym.harness.mock_training",
                "--run-dir",
                "{run_dir}",
                "--mode",
                "trend",
                "--steps",
                "300",
                "--delay",
                "0.02",
            ],
            authority="supervisor",
            output_root=output_root,
            require_rtx4090=False,
            mock=True,
            trend_min_iteration=3,
        )
        deadline = time.time() + 10
        snapshot = None
        while time.time() < deadline:
            snapshot = collect_run(
                launched["run_dir"], stale_seconds=30, trend_min_iteration=3
            )
            if snapshot["trend"]["candidate_stop"]:
                break
            if snapshot["state"]["state"] in TERMINAL_STATES:
                raise HarnessError("Trend fixture exited before tracker decision")
            time.sleep(0.05)
        if snapshot is None or not snapshot["trend"]["candidate_stop"]:
            raise HarnessError("Trend smoke never produced a stop candidate")
        decision = {
            "confidence": snapshot["trend"]["confidence"],
            "evidence": snapshot["trend"]["evidence"],
            "alternatives_considered": ["continue to next checkpoint"],
            "continuation_cost": "mock-only",
            "evidence_window": "mock rolling window",
            "last_usable_checkpoint": snapshot["checkpoints"][-1]["path"],
        }
        stopped = stop_run(
            launched["run_dir"],
            kind="trend",
            reason="smoke plateau with tracking and gait corroboration",
            authority="tracker",
            decision=decision,
            grace_seconds=3,
        )
        if stopped["state"]["state"] != RunState.STOPPED_TREND.value:
            raise HarnessError("Trend stop ended in {}".format(stopped["state"]["state"]))
        states, versions = _event_states(launched["run_dir"])
        if versions != sorted(set(versions)):
            raise HarnessError("Non-monotonic state versions in trend smoke")
        results["trend"] = {
            "state": stopped["state"]["state"],
            "run_id": launched["run_id"],
            "events": states,
            "analysis_requested": stopped["state"]["analysis_requested"],
            "trend": snapshot["trend"],
            "exit": stopped["exit"],
        }
        return results
    finally:
        if temporary is not None:
            temporary.cleanup()


def _build_parser():
    parser = argparse.ArgumentParser(description="Managed locomotion-training harness")
    subparsers = parser.add_subparsers(dest="command_name", required=True)

    subparsers.add_parser("gpu-status")

    validate_p0 = subparsers.add_parser("validate-p0")
    validate_p0.add_argument("--run-dir", required=True)
    validate_p0.add_argument(
        "--profile",
        choices=(
            "teacher45",
            "official_wim_a1_rough_v1",
            "comparison_b1",
            "comparison_b2",
        ),
        default="teacher45",
    )

    launch = subparsers.add_parser("launch")
    launch.add_argument("--authority", required=True)
    launch.add_argument("--phase", required=True)
    launch.add_argument("--experiment", required=True)
    launch.add_argument("--label", required=True)
    launch.add_argument("--parent-run-id")
    launch.add_argument("--output-root")
    launch.add_argument("--repo-root", default=str(Path.cwd()))
    launch.add_argument("--morai-running", action="store_true")
    launch.add_argument("--trend-min-iteration", type=int, default=250)
    launch.add_argument("--mock", action="store_true", help=argparse.SUPPRESS)
    launch.add_argument("argv", nargs=argparse.REMAINDER)

    for name in ("collect", "status"):
        command = subparsers.add_parser(name)
        command.add_argument("--run-dir", required=True)
        command.add_argument("--stale-seconds", type=float, default=600)
        command.add_argument("--trend-min-iteration", type=int, default=250)

    watch = subparsers.add_parser("watch")
    watch.add_argument("--run-dir", required=True)
    watch.add_argument("--poll-seconds", type=float, default=180)
    watch.add_argument("--stale-seconds", type=float, default=600)
    watch.add_argument("--trend-min-iteration", type=int, default=250)
    watch.add_argument("--event-driven", action="store_true")
    watch.add_argument("--checkpoint-interval", type=int, default=0)

    stop = subparsers.add_parser("stop")
    stop.add_argument("--run-dir", required=True)
    stop.add_argument("--authority", required=True)
    stop.add_argument("--kind", choices=("trend", "operator", "hard"), required=True)
    stop.add_argument("--reason", required=True)
    stop.add_argument("--decision-json")
    stop.add_argument("--grace-seconds", type=float, default=30)

    claim = subparsers.add_parser("claim-analysis")
    claim.add_argument("--run-dir", required=True)

    reopen = subparsers.add_parser("reopen-analysis")
    reopen.add_argument("--run-dir", required=True)
    reopen.add_argument("--authority", required=True)
    reopen.add_argument("--expected-state-version", required=True, type=int)
    reopen.add_argument("--expected-plan-sha256", required=True)
    reopen.add_argument("--reason", required=True)

    resume_blocked = subparsers.add_parser("resume-blocked-analysis")
    resume_blocked.add_argument("--run-dir", required=True)
    resume_blocked.add_argument("--authority", required=True)
    resume_blocked.add_argument("--expected-state-version", required=True, type=int)
    resume_blocked.add_argument("--expected-plan-sha256", required=True)
    resume_blocked.add_argument("--scope-decision", required=True)

    route = subparsers.add_parser("route-analysis")
    route.add_argument("--run-dir", required=True)
    route.add_argument("--authority", required=True)
    route.add_argument("--disposition", choices=("modify", "ready"), required=True)
    route.add_argument("--expected-state-version", type=int)

    verify = subparsers.add_parser("begin-verification")
    verify.add_argument("--run-dir", required=True)
    verify.add_argument("--authority", required=True)

    finish = subparsers.add_parser("finish-modification")
    finish.add_argument("--run-dir", required=True)
    finish.add_argument("--authority", required=True)

    stage_error = subparsers.add_parser("record-stage-error")
    stage_error.add_argument("--run-dir", required=True)
    stage_error.add_argument("--authority", required=True)
    stage_error.add_argument("--reason", required=True)

    blocked = subparsers.add_parser("mark-blocked")
    blocked.add_argument("--run-dir", required=True)
    blocked.add_argument("--authority", required=True)
    blocked.add_argument("--reason", required=True)

    supervise_parser = subparsers.add_parser("_supervise", help=argparse.SUPPRESS)
    supervise_parser.add_argument("--run-dir", required=True)

    smoke = subparsers.add_parser("smoke")
    smoke.add_argument("--output-root")
    return parser


def main(argv=None):
    args = _build_parser().parse_args(argv)
    try:
        if args.command_name == "gpu-status":
            _print(gpu_snapshot())
            return 0
        if args.command_name == "validate-p0":
            report = validate_p0_gate(args.run_dir, profile=args.profile)
            _print(report)
            return 0 if report["valid"] else 2
        if args.command_name == "launch":
            command = list(args.argv)
            if command and command[0] == "--":
                command = command[1:]
            result = launch_run(
                repo_root=args.repo_root,
                phase=args.phase,
                experiment=args.experiment,
                label=args.label,
                command=command,
                authority=args.authority,
                parent_run_id=args.parent_run_id,
                output_root=args.output_root,
                require_rtx4090=not args.mock,
                mock=args.mock,
                morai_reported_running=args.morai_running,
                trend_min_iteration=args.trend_min_iteration,
            )
            _print(result)
            return 0
        if args.command_name in {"collect", "status"}:
            _print(
                collect_run(
                    args.run_dir,
                    stale_seconds=args.stale_seconds,
                    trend_min_iteration=args.trend_min_iteration,
                )
            )
            return 0
        if args.command_name == "watch":
            while True:
                snapshot = collect_run(
                    args.run_dir,
                    stale_seconds=args.stale_seconds,
                    trend_min_iteration=args.trend_min_iteration,
                )
                if args.event_driven:
                    event = _monitor_event(
                        args.run_dir,
                        snapshot,
                        checkpoint_interval=max(args.checkpoint_interval, 0),
                    )
                    if event is not None:
                        _print(event)
                        return 0
                else:
                    _print(snapshot)
                    sys.stdout.flush()
                if snapshot["state"]["state"] in TERMINAL_STATES:
                    return 0
                time.sleep(max(args.poll_seconds, 1.0))
        if args.command_name == "stop":
            decision = {}
            if args.decision_json:
                decision = json.loads(Path(args.decision_json).read_text(encoding="utf-8"))
            _print(
                stop_run(
                    args.run_dir,
                    kind=args.kind,
                    reason=args.reason,
                    authority=args.authority,
                    decision=decision,
                    grace_seconds=args.grace_seconds,
                )
            )
            return 0
        if args.command_name == "claim-analysis":
            result = RunStore(args.run_dir).claim_analysis()
            _print(result)
            return 0 if result["claimed"] else 2
        if args.command_name == "reopen-analysis":
            _print(
                reopen_analysis(
                    args.run_dir,
                    args.authority,
                    args.expected_state_version,
                    args.expected_plan_sha256,
                    args.reason,
                )
            )
            return 0
        if args.command_name == "resume-blocked-analysis":
            _print(
                resume_blocked_analysis(
                    args.run_dir,
                    args.authority,
                    args.expected_state_version,
                    args.expected_plan_sha256,
                    args.scope_decision,
                )
            )
            return 0
        if args.command_name == "route-analysis":
            _print(
                route_analysis(
                    args.run_dir,
                    args.disposition,
                    args.authority,
                    expected_state_version=args.expected_state_version,
                )
            )
            return 0
        if args.command_name == "begin-verification":
            _print(begin_verification(args.run_dir, args.authority))
            return 0
        if args.command_name == "finish-modification":
            _print(finish_modification(args.run_dir, args.authority))
            return 0
        if args.command_name == "record-stage-error":
            _print(record_stage_error(args.run_dir, args.reason, args.authority))
            return 0
        if args.command_name == "mark-blocked":
            _print(mark_blocked(args.run_dir, args.reason, args.authority))
            return 0
        if args.command_name == "_supervise":
            return supervise(args.run_dir)
        if args.command_name == "smoke":
            _print(smoke_loop(args.output_root))
            return 0
    except (HarnessError, OSError, ValueError, json.JSONDecodeError) as exc:
        print(json.dumps({"error": str(exc)}, sort_keys=True), file=sys.stderr)
        return 1
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
