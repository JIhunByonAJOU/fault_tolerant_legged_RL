"""Process-safe state and monitoring for long-running Isaac Gym experiments."""

import contextlib
import datetime as dt
import fcntl
import hashlib
import json
import math
import os
import re
import signal
import shutil
import statistics
import subprocess
import sys
import time
import uuid
from enum import Enum
from pathlib import Path


class HarnessError(RuntimeError):
    """Raised when a managed-run invariant would be violated."""


class RunState(str, Enum):
    PREFLIGHT = "PREFLIGHT"
    RUNNING = "RUNNING"
    STOPPING = "STOPPING"
    COMPLETED = "COMPLETED"
    STOPPED_TREND = "STOPPED_TREND"
    STOPPED_OPERATOR = "STOPPED_OPERATOR"
    ERROR = "ERROR"
    ANALYZING = "ANALYZING"
    MODIFYING = "MODIFYING"
    VERIFYING = "VERIFYING"
    READY = "READY"
    BLOCKED = "BLOCKED"


TERMINAL_STATES = {
    RunState.COMPLETED.value,
    RunState.STOPPED_TREND.value,
    RunState.STOPPED_OPERATOR.value,
    RunState.ERROR.value,
    RunState.BLOCKED.value,
}

ANALYZABLE_STATES = {
    RunState.COMPLETED.value,
    RunState.STOPPED_TREND.value,
    RunState.STOPPED_OPERATOR.value,
    RunState.ERROR.value,
}

_ALLOWED_TRANSITIONS = {
    RunState.PREFLIGHT.value: {RunState.RUNNING.value, RunState.ERROR.value, RunState.BLOCKED.value},
    RunState.RUNNING.value: {RunState.STOPPING.value, RunState.COMPLETED.value, RunState.ERROR.value},
    RunState.STOPPING.value: {
        RunState.STOPPED_TREND.value,
        RunState.STOPPED_OPERATOR.value,
        RunState.ERROR.value,
    },
    RunState.COMPLETED.value: {RunState.ANALYZING.value, RunState.BLOCKED.value},
    RunState.STOPPED_TREND.value: {RunState.ANALYZING.value, RunState.BLOCKED.value},
    RunState.STOPPED_OPERATOR.value: {RunState.ANALYZING.value, RunState.BLOCKED.value},
    RunState.ERROR.value: {RunState.ANALYZING.value, RunState.BLOCKED.value},
    RunState.ANALYZING.value: {
        RunState.MODIFYING.value,
        RunState.READY.value,
        RunState.ERROR.value,
        RunState.BLOCKED.value,
    },
    RunState.MODIFYING.value: {
        RunState.VERIFYING.value,
        RunState.ERROR.value,
        RunState.BLOCKED.value,
    },
    RunState.VERIFYING.value: {
        RunState.MODIFYING.value,
        RunState.READY.value,
        RunState.ERROR.value,
        RunState.BLOCKED.value,
    },
    RunState.READY.value: {RunState.BLOCKED.value},
    RunState.BLOCKED.value: set(),
}

_ACTIVE_STATES = {RunState.PREFLIGHT.value, RunState.RUNNING.value, RunState.STOPPING.value}
_SLUG_PATTERN = re.compile(r"[^a-z0-9._-]+")
_HARD_ERROR_PATTERNS = (
    re.compile(r"\b(?:nan|inf)\b", re.IGNORECASE),
    re.compile(r"CUDA out of memory", re.IGNORECASE),
    re.compile(r"Traceback \(most recent call last\):"),
    re.compile(r"(?:RuntimeError|ValueError|AssertionError):"),
)
_LOCAL_SUPERVISOR_HANDLES = {}
_HEARTBEAT_INTERVAL_SECONDS = 2.0
_HEARTBEAT_FRESH_SECONDS = 10.0
_LIVENESS_GRACE_SECONDS = 15.0


def _now_iso():
    return dt.datetime.now().astimezone().isoformat(timespec="seconds")


def _read_boot_id():
    path = Path("/proc/sys/kernel/random/boot_id")
    try:
        return path.read_text(encoding="utf-8").strip()
    except OSError:
        return None


def _slug(value, fallback):
    clean = _SLUG_PATTERN.sub("-", str(value).strip().lower()).strip("-._")
    return clean or fallback


def _read_json(path, default=None):
    path = Path(path)
    if not path.exists():
        return default
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _atomic_write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp-" + uuid.uuid4().hex)
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(str(temporary), str(path))


def _append_jsonl(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, sort_keys=True) + "\n")
        handle.flush()


def _preserve_digest_addressed_bytes(path, content, digest):
    """Atomically create a read-only digest-addressed archive without overwriting it."""
    path = Path(path)
    actual_digest = hashlib.sha256(content).hexdigest()
    if actual_digest != digest or path.name != digest + ".json":
        raise HarnessError(
            "Digest-addressed archive identity mismatch: {} ({})".format(path, actual_digest)
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != content:
            raise HarnessError("Digest-addressed archive content mismatch: {}".format(path))
        return

    temporary = path.with_name(path.name + ".tmp-" + uuid.uuid4().hex)
    try:
        with temporary.open("xb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(str(temporary), 0o444)
        try:
            os.link(str(temporary), str(path))
        except FileExistsError:
            if path.read_bytes() != content:
                raise HarnessError("Digest-addressed archive content mismatch: {}".format(path))
        directory_fd = os.open(str(path.parent), os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


class RunStore:
    """Atomic state/event access shared by the agent supervisor and tracker."""

    def __init__(self, run_dir):
        self.run_dir = Path(run_dir).resolve()
        self.manifest_path = self.run_dir / "manifest.json"
        self.state_path = self.run_dir / "state.json"
        self.events_path = self.run_dir / "events.jsonl"
        self.lock_path = self.run_dir / ".state.lock"

    @contextlib.contextmanager
    def locked(self):
        self.run_dir.mkdir(parents=True, exist_ok=True)
        with self.lock_path.open("a+") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    def initialize(self, manifest):
        with self.locked():
            if self.manifest_path.exists() or self.state_path.exists():
                raise HarnessError("Run directory already contains managed state: {}".format(self.run_dir))
            _atomic_write_json(self.manifest_path, manifest)
            state = {
                "run_id": manifest["run_id"],
                "state": RunState.PREFLIGHT.value,
                "state_version": 1,
                "updated_at": _now_iso(),
                "analysis_requested": False,
                "analysis_dispatched_version": None,
                "stop": None,
                "last_error": None,
            }
            _atomic_write_json(self.state_path, state)
            _append_jsonl(
                self.events_path,
                {"at": state["updated_at"], "from": None, "to": state["state"], "version": 1},
            )
        return state

    def manifest(self):
        manifest = _read_json(self.manifest_path)
        if manifest is None:
            raise HarnessError("Missing manifest: {}".format(self.manifest_path))
        return manifest

    def update_manifest(self, **updates):
        with self.locked():
            manifest = self.manifest()
            manifest.update(updates)
            _atomic_write_json(self.manifest_path, manifest)
        return manifest

    def state(self):
        state = _read_json(self.state_path)
        if state is None:
            raise HarnessError("Missing state: {}".format(self.state_path))
        return state

    def transition(
        self,
        target,
        reason=None,
        expected_states=None,
        expected_state_version=None,
        fail_on_mismatch=False,
        **updates
    ):
        target = target.value if isinstance(target, RunState) else str(target)
        with self.locked():
            state = self.state()
            previous = state["state"]
            if expected_states is not None and previous not in set(expected_states):
                if fail_on_mismatch:
                    raise HarnessError(
                        "Managed-run state mismatch: expected {}, found {}".format(
                            sorted(set(expected_states)), previous
                        )
                    )
                return state
            version = int(state.get("state_version", 0))
            if expected_state_version is not None and version != int(expected_state_version):
                if fail_on_mismatch:
                    raise HarnessError(
                        "Managed-run state version mismatch: expected {}, found {}".format(
                            expected_state_version, version
                        )
                    )
                return state
            if previous == target:
                return state
            if target not in _ALLOWED_TRANSITIONS.get(previous, set()):
                raise HarnessError("Illegal managed-run transition: {} -> {}".format(previous, target))
            state.update(updates)
            state["state"] = target
            state["state_version"] = int(state.get("state_version", 0)) + 1
            state["updated_at"] = _now_iso()
            if target in TERMINAL_STATES:
                state["analysis_requested"] = True
            _atomic_write_json(self.state_path, state)
            _append_jsonl(
                self.events_path,
                {
                    "at": state["updated_at"],
                    "from": previous,
                    "to": target,
                    "version": state["state_version"],
                    "reason": reason,
                },
            )
        return state

    def claim_analysis(self):
        """Atomically claim one terminal-state analysis dispatch."""
        with self.locked():
            state = self.state()
            version = int(state.get("state_version", 0))
            if state["state"] not in ANALYZABLE_STATES:
                return {"claimed": False, "reason": "run is not terminal", "state": state}
            if not state.get("analysis_requested"):
                return {"claimed": False, "reason": "analysis not requested", "state": state}
            origin = state["state"]
            mode = "incident" if origin == RunState.ERROR.value else "experiment"
            state["analysis_requested"] = False
            state["analysis_dispatched_version"] = version
            state["analysis_claimed_at"] = _now_iso()
            state["analysis_origin"] = {
                "state": origin,
                "state_version": version,
                "mode": mode,
            }
            state["state"] = RunState.ANALYZING.value
            state["state_version"] = version + 1
            state["updated_at"] = state["analysis_claimed_at"]
            _atomic_write_json(self.state_path, state)
            _append_jsonl(
                self.events_path,
                {
                    "at": state["analysis_claimed_at"],
                    "from": origin,
                    "to": RunState.ANALYZING.value,
                    "version": state["state_version"],
                    "reason": "analysis claimed exactly once",
                    "analysis_mode": mode,
                },
            )
        return {"claimed": True, "mode": mode, "state": state}

    def reopen_analysis(
        self,
        authority,
        expected_state_version,
        expected_plan_sha256,
        reason,
    ):
        """Atomically reopen one superseded READY analysis without reclaiming it."""
        if authority != "supervisor":
            raise HarnessError("Only authority=supervisor may reopen an analysis")
        if not str(reason).strip():
            raise HarnessError("reopen-analysis requires a non-empty reason")
        if not re.fullmatch(r"[0-9a-f]{64}", str(expected_plan_sha256)):
            raise HarnessError("expected_plan_sha256 must be a lowercase SHA-256 digest")

        plan_path = self.run_dir / "analysis_plan.json"
        with self.locked():
            state = self.state()
            if state["state"] != RunState.READY.value:
                raise HarnessError(
                    "reopen-analysis state mismatch: expected READY, found {}".format(
                        state["state"]
                    )
                )
            version = int(state.get("state_version", 0))
            if version != int(expected_state_version):
                raise HarnessError(
                    "reopen-analysis state version mismatch: expected {}, found {}".format(
                        expected_state_version, version
                    )
                )
            prior_plan = state.get("analysis_plan")
            prior_digest = prior_plan.get("sha256") if isinstance(prior_plan, dict) else None
            if prior_digest != expected_plan_sha256:
                raise HarnessError(
                    "reopen-analysis prior plan digest mismatch: expected {}, found {}".format(
                        expected_plan_sha256, prior_digest
                    )
                )

            try:
                replacement_bytes = plan_path.read_bytes()
                replacement_plan = json.loads(replacement_bytes.decode("utf-8"))
            except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise HarnessError(
                    "Missing or invalid analysis_plan.json: {} ({})".format(plan_path, exc)
                )
            _validate_analysis_plan_value(replacement_plan, plan_path)
            replacement_digest = hashlib.sha256(replacement_bytes).hexdigest()
            if replacement_digest == prior_digest:
                raise HarnessError("reopen-analysis replacement plan must differ from prior plan")

            previous = dict(state)
            reopened_at = _now_iso()
            state["superseded_readiness"] = {
                "reason": str(reason).strip(),
                "record": previous,
                "replacement_analysis_plan": {
                    "path": str(plan_path),
                    "sha256": replacement_digest,
                },
            }
            state["state"] = RunState.ANALYZING.value
            state["state_version"] = version + 1
            state["updated_at"] = reopened_at
            state["analysis_requested"] = False
            _atomic_write_json(self.state_path, state)
            _append_jsonl(
                self.events_path,
                {
                    "at": reopened_at,
                    "from": RunState.READY.value,
                    "to": RunState.ANALYZING.value,
                    "version": state["state_version"],
                    "reason": str(reason).strip(),
                    "prior_plan_sha256": prior_digest,
                    "replacement_plan_sha256": replacement_digest,
                },
            )
        return state

    def resume_blocked_analysis(
        self,
        authority,
        expected_state_version,
        expected_plan_sha256,
        scope_decision,
    ):
        """Resume one explicitly re-scoped BLOCKED analysis with strict CAS checks."""
        if authority != "supervisor":
            raise HarnessError("Only authority=supervisor may resume a blocked analysis")
        scope_decision = str(scope_decision).strip()
        if not scope_decision:
            raise HarnessError("resume-blocked-analysis requires a non-empty scope decision")
        expected_plan_sha256 = str(expected_plan_sha256)
        if not re.fullmatch(r"[0-9a-f]{64}", expected_plan_sha256):
            raise HarnessError("expected_plan_sha256 must be a lowercase SHA-256 digest")

        plan_path = self.run_dir / "analysis_plan.json"
        with self.locked():
            state = self.state()
            if state["state"] != RunState.BLOCKED.value:
                raise HarnessError(
                    "resume-blocked-analysis state mismatch: expected BLOCKED, found {}".format(
                        state["state"]
                    )
                )
            version = int(state.get("state_version", 0))
            if version != int(expected_state_version):
                raise HarnessError(
                    "resume-blocked-analysis state version mismatch: expected {}, found {}".format(
                        expected_state_version, version
                    )
                )

            prior_plan = state.get("analysis_plan")
            recorded_digest = (
                prior_plan.get("sha256") if isinstance(prior_plan, dict) else None
            )
            if recorded_digest is not None and recorded_digest != expected_plan_sha256:
                raise HarnessError(
                    "resume-blocked-analysis recorded plan digest mismatch: expected {}, found {}".format(
                        expected_plan_sha256, recorded_digest
                    )
                )
            try:
                prior_plan_bytes = plan_path.read_bytes()
            except OSError as exc:
                raise HarnessError(
                    "Missing prior analysis_plan.json: {} ({})".format(plan_path, exc)
                )
            actual_digest = hashlib.sha256(prior_plan_bytes).hexdigest()
            if actual_digest != expected_plan_sha256:
                raise HarnessError(
                    "resume-blocked-analysis current plan digest mismatch: expected {}, found {}".format(
                        expected_plan_sha256, actual_digest
                    )
                )

            archive_path = self.run_dir / "analysis-history" / (
                expected_plan_sha256 + ".json"
            )
            _preserve_digest_addressed_bytes(
                archive_path, prior_plan_bytes, expected_plan_sha256
            )

            previous = dict(state)
            resumed_at = _now_iso()
            prior_plan_record = {
                "path": str(plan_path),
                "sha256": actual_digest,
                "archive_path": str(archive_path),
            }
            state["blocked_analysis_resume"] = {
                "scope_decision": scope_decision,
                "prior_blocked_reason": previous.get("blocked_reason"),
                "prior_state": previous["state"],
                "prior_state_version": version,
                "prior_analysis_plan": prior_plan_record,
                "prior_state_record": previous,
            }
            state["state"] = RunState.ANALYZING.value
            state["state_version"] = version + 1
            state["updated_at"] = resumed_at
            state["analysis_requested"] = False
            _atomic_write_json(self.state_path, state)
            _append_jsonl(
                self.events_path,
                {
                    "at": resumed_at,
                    "from": RunState.BLOCKED.value,
                    "to": RunState.ANALYZING.value,
                    "version": state["state_version"],
                    "reason": "explicit user scope decision resumed blocked analysis",
                    "scope_decision": scope_decision,
                    "prior_blocked_reason": previous.get("blocked_reason"),
                    "prior_plan_path": str(plan_path),
                    "prior_plan_sha256": actual_digest,
                    "prior_plan_archive_path": str(archive_path),
                },
            )
        return state


def _validated_json_object(path, description):
    path = Path(path)
    value = _read_json(path)
    if not isinstance(value, dict) or not value:
        raise HarnessError("Missing or invalid {}: {}".format(description, path))
    return value


def _validate_analysis_plan_value(plan, path):
    if not isinstance(plan, dict) or not plan:
        raise HarnessError("Missing or invalid analysis_plan.json: {}".format(path))
    required = {
        "analysis_mode",
        "observed_outcome",
        "primary_hypothesis",
        "exact_proposed_changes",
        "expected_metric_response",
        "disconfirming_result",
        "validation_commands",
        "next_run_budget",
    }
    missing = sorted(required.difference(plan))
    if missing:
        raise HarnessError("analysis_plan.json is missing keys: {}".format(", ".join(missing)))
    return plan


def _validated_analysis_plan(path):
    return _validate_analysis_plan_value(
        _validated_json_object(path, "analysis_plan.json"), Path(path)
    )


def reopen_analysis(
    run_dir,
    authority,
    expected_state_version,
    expected_plan_sha256,
    reason,
):
    return RunStore(run_dir).reopen_analysis(
        authority=authority,
        expected_state_version=expected_state_version,
        expected_plan_sha256=expected_plan_sha256,
        reason=reason,
    )


def resume_blocked_analysis(
    run_dir,
    authority,
    expected_state_version,
    expected_plan_sha256,
    scope_decision,
):
    return RunStore(run_dir).resume_blocked_analysis(
        authority=authority,
        expected_state_version=expected_state_version,
        expected_plan_sha256=expected_plan_sha256,
        scope_decision=scope_decision,
    )


def route_analysis(run_dir, disposition, authority, expected_state_version=None):
    """Supervisor-approved routing from an analysis plan to modification or READY."""
    if authority != "supervisor":
        raise HarnessError("Only authority=supervisor may approve an analysis route")
    if disposition not in {"modify", "ready"}:
        raise HarnessError("Unknown analysis disposition: {}".format(disposition))
    store = RunStore(run_dir)
    plan_path = store.run_dir / "analysis_plan.json"
    _validated_analysis_plan(plan_path)
    digest = hashlib.sha256(plan_path.read_bytes()).hexdigest()
    target = RunState.MODIFYING if disposition == "modify" else RunState.READY
    return store.transition(
        target,
        reason="supervisor approved analysis disposition={}".format(disposition),
        expected_states={RunState.ANALYZING.value},
        expected_state_version=expected_state_version,
        fail_on_mismatch=True,
        analysis_plan={"path": str(plan_path), "sha256": digest, "disposition": disposition},
    )


def begin_verification(run_dir, authority):
    if authority != "modifier":
        raise HarnessError("Only authority=modifier may begin modification verification")
    store = RunStore(run_dir)
    return store.transition(
        RunState.VERIFYING,
        reason="modifier began two-pass verification",
        expected_states={RunState.MODIFYING.value},
    )


def finish_modification(run_dir, authority):
    if authority != "modifier":
        raise HarnessError("Only authority=modifier may finish modification verification")
    store = RunStore(run_dir)
    report_path = store.run_dir / "modification_report.json"
    report = _validated_json_object(report_path, "modification_report.json")
    if not isinstance(report.get("production_launch_ready"), bool):
        raise HarnessError("modification_report.json needs boolean production_launch_ready")
    digest = hashlib.sha256(report_path.read_bytes()).hexdigest()
    report_record = {"path": str(report_path), "sha256": digest}
    if report["production_launch_ready"]:
        return store.transition(
            RunState.READY,
            reason="modifier verification passed",
            expected_states={RunState.VERIFYING.value},
            modification_report=report_record,
        )
    return store.transition(
        RunState.ERROR,
        reason="modifier verification failed",
        expected_states={RunState.VERIFYING.value},
        modification_report=report_record,
        last_error={"kind": "modification_verification", "report": report_record},
    )


def record_stage_error(run_dir, reason, authority):
    if authority != "supervisor":
        raise HarnessError("Only authority=supervisor may record an agent-stage error")
    store = RunStore(run_dir)
    return store.transition(
        RunState.ERROR,
        reason=reason,
        expected_states={
            RunState.ANALYZING.value,
            RunState.MODIFYING.value,
            RunState.VERIFYING.value,
        },
        last_error={"kind": "agent_stage", "detail": reason},
    )


def mark_blocked(run_dir, reason, authority):
    if authority != "supervisor":
        raise HarnessError("Only authority=supervisor may mark a managed run BLOCKED")
    store = RunStore(run_dir)
    return store.transition(
        RunState.BLOCKED,
        reason=reason,
        expected_states=set(_ALLOWED_TRANSITIONS).difference({RunState.BLOCKED.value}),
        blocked_reason=reason,
    )


def _repo_snapshot(repo_root):
    repo_root = str(Path(repo_root).resolve())
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo_root, text=True, stderr=subprocess.DEVNULL
        ).strip()
        status = subprocess.check_output(
            ["git", "status", "--porcelain=v1"], cwd=repo_root, text=True
        )
        diff = subprocess.check_output(
            ["git", "diff", "--binary", "HEAD"], cwd=repo_root
        )
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None, "dirty_files": [], "diff_sha256": None}
    dirty_files = [line[3:] for line in status.splitlines() if len(line) > 3]
    identity = diff + status.encode("utf-8")
    return {
        "commit": commit,
        "dirty": bool(status.strip()),
        "dirty_files": dirty_files,
        "diff_sha256": hashlib.sha256(identity).hexdigest(),
    }


def gpu_snapshot():
    """Return a small 4090 preflight record without requiring NVML bindings."""
    command = [
        "nvidia-smi",
        "--query-gpu=index,name,memory.total,memory.used,utilization.gpu,temperature.gpu,pstate",
        "--format=csv,noheader,nounits",
    ]
    try:
        output = subprocess.check_output(command, text=True, stderr=subprocess.STDOUT, timeout=10)
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        return {"available": False, "error": str(exc), "gpus": []}
    gpus = []
    for line in output.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 7:
            continue
        gpus.append(
            {
                "index": int(fields[0]),
                "name": fields[1],
                "memory_total_mib": int(fields[2]),
                "memory_used_mib": int(fields[3]),
                "utilization_percent": int(fields[4]),
                "temperature_c": int(fields[5]),
                "pstate": fields[6],
            }
        )
    processes = []
    try:
        process_output = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,process_name,used_gpu_memory",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=10,
        )
        for line in process_output.splitlines():
            fields = [field.strip() for field in line.split(",", 2)]
            if len(fields) != 3 or not fields[0].isdigit():
                continue
            try:
                used_memory_mib = int(fields[2])
            except ValueError:
                used_memory_mib = None
            processes.append(
                {
                    "pid": int(fields[0]),
                    "process_name": fields[1],
                    "used_memory_mib": used_memory_mib,
                }
            )
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        pass
    return {
        "available": bool(gpus),
        "captured_at": _now_iso(),
        "gpus": gpus,
        "compute_processes": processes,
    }


def disk_snapshot(path):
    usage = shutil.disk_usage(str(Path(path).resolve()))
    gib = float(1024 ** 3)
    return {
        "captured_at": _now_iso(),
        "path": str(Path(path).resolve()),
        "total_gib": usage.total / gib,
        "used_gib": usage.used / gib,
        "free_gib": usage.free / gib,
    }


def _proc_start_ticks(pid):
    try:
        fields = Path("/proc/{}/stat".format(pid)).read_text(encoding="utf-8").split()
        return int(fields[21])
    except (OSError, ValueError, IndexError):
        return None


def _process_alive(pid, expected_start_ticks=None):
    try:
        stat_fields = Path("/proc/{}/stat".format(pid)).read_text(encoding="utf-8").split()
    except OSError:
        return False
    if len(stat_fields) < 22 or stat_fields[2] == "Z":
        return False
    if expected_start_ticks is not None and int(stat_fields[21]) != int(expected_start_ticks):
        return False
    return True


def _supervisor_heartbeat(manifest, heartbeat, now=None):
    """Return heartbeat age only when it belongs to this exact supervisor."""
    if not isinstance(heartbeat, dict):
        return None
    expected = {
        "run_id": manifest.get("run_id"),
        "supervisor_pid": int(manifest.get("supervisor_pid") or 0),
        "supervisor_start_ticks": manifest.get("supervisor_start_ticks"),
    }
    try:
        actual = {
            "run_id": heartbeat.get("run_id"),
            "supervisor_pid": int(heartbeat.get("supervisor_pid") or 0),
            "supervisor_start_ticks": heartbeat.get("supervisor_start_ticks"),
        }
        wall_time = float(heartbeat["wall_time_unix"])
    except (KeyError, TypeError, ValueError):
        return None
    if actual != expected:
        return None
    return max(0.0, float(time.time() if now is None else now) - wall_time)


def _clear_liveness_probe(run_dir):
    try:
        (Path(run_dir) / "liveness_probe.json").unlink()
    except FileNotFoundError:
        pass


def _progress_signature(run_dir):
    signature = {}
    for name in ("metrics.jsonl", "console.log"):
        path = Path(run_dir) / name
        if path.exists():
            stat = path.stat()
            signature[name] = {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns}
    return signature


_PRODUCTION_ENTRYPOINT_TASKS = {
    "legged_gym/scripts/train.py": {
        "a1_limping_base",
        "a1_limping_base_v2",
        "a1_limping_base_wim",
        "a1_official_wim_teacher243_failure",
    },
    "legged_gym/scripts/train_official_wim_managed.py": {"a1_official_wim_rough"},
}
_SHELL_COMMAND_TOKENS = {"bash", "sh", "zsh"}
_SHELL_CONTROL_TOKENS = {"-c", "|", "||", "&&", ";", "<", ">", ">>", "2>", "2>&1"}
_MANAGED_IDENTITY_PLACEHOLDERS = {
    "{run_dir}": "run_dir",
    "{run_id}": "run_id",
    "{run_name}": "run_id",
}


def _single_option_value(command, option):
    occurrences = []
    prefix = option + "="
    for index, token in enumerate(command):
        if token == option:
            if index + 1 >= len(command) or str(command[index + 1]).startswith("--"):
                raise HarnessError("{} requires a value".format(option))
            occurrences.append(str(command[index + 1]))
        elif str(token).startswith(prefix):
            value = str(token)[len(prefix) :]
            if not value:
                raise HarnessError("{} requires a value".format(option))
            occurrences.append(value)
    if len(occurrences) > 1:
        raise HarnessError("Managed launch rejects duplicate {} arguments".format(option))
    return occurrences[0] if occurrences else None


def _validate_production_command(command, repo_root):
    command = [str(token) for token in command]
    if "--headless" not in command:
        raise HarnessError("Managed production training must be headless")
    if any(
        Path(token).name in _SHELL_COMMAND_TOKENS or token in _SHELL_CONTROL_TOKENS
        for token in command
    ):
        raise HarnessError("Managed launch accepts an argument vector, not a shell command")

    repo_root = Path(repo_root).resolve()
    script_tokens = [token for token in command if Path(token).suffix == ".py"]
    if len(script_tokens) != 1:
        raise HarnessError("Managed production launch requires exactly one script argv token")
    script_path = Path(script_tokens[0])
    if not script_path.is_absolute():
        script_path = repo_root / script_path
    script_path = script_path.resolve()

    allowed_by_path = {
        (repo_root / relative).resolve(): tasks
        for relative, tasks in _PRODUCTION_ENTRYPOINT_TASKS.items()
    }
    allowed_tasks = allowed_by_path.get(script_path)
    if allowed_tasks is None:
        raise HarnessError("Production entrypoint is not allowlisted: {}".format(script_tokens[0]))

    task = _single_option_value(command, "--task")
    if task is None:
        raise HarnessError("Managed production launch requires exactly one --task")
    if task not in allowed_tasks:
        raise HarnessError(
            "Production entrypoint/task pair is not allowlisted: {} / {}".format(
                script_tokens[0], task
            )
        )


def _materialize_managed_identity(command, run_dir, run_id):
    values = {"run_dir": str(Path(run_dir).resolve()), "run_id": str(run_id)}
    materialized = []
    for original in command:
        token = str(original)
        key = _MANAGED_IDENTITY_PLACEHOLDERS.get(token)
        if key is not None:
            token = values[key]
        else:
            for option in ("--log_dir", "--run_name"):
                prefix = option + "="
                if token.startswith(prefix):
                    value = token[len(prefix) :]
                    value_key = _MANAGED_IDENTITY_PLACEHOLDERS.get(value)
                    if value_key is not None:
                        token = prefix + values[value_key]
                    break
        if "{" in token or "}" in token:
            raise HarnessError("Managed launch rejects unknown, embedded, or unresolved placeholders")
        materialized.append(token)
    return materialized


def _inject_managed_identity(command, run_dir, run_id):
    command = list(command)
    expected = {
        "--log_dir": str(Path(run_dir).resolve()),
        "--run_name": str(run_id),
    }
    for option, expected_value in expected.items():
        value = _single_option_value(command, option)
        if value is None:
            command.extend([option, expected_value])
        elif value != expected_value:
            raise HarnessError(
                "{} must equal managed identity {}; found {}".format(
                    option, expected_value, value
                )
            )

    for option, expected_value in expected.items():
        if _single_option_value(command, option) != expected_value:
            raise HarnessError("Final managed identity invariant failed for {}".format(option))
    if any("{" in str(token) or "}" in str(token) for token in command):
        raise HarnessError("Final managed command contains an unresolved placeholder")
    return command


def launch_run(
    repo_root,
    phase,
    experiment,
    label,
    command,
    authority,
    parent_run_id=None,
    output_root=None,
    require_rtx4090=True,
    mock=False,
    morai_reported_running=False,
    trend_min_iteration=250,
):
    """Launch a command under a durable supervisor and return its run directory."""
    if authority != "supervisor":
        raise HarnessError("Only authority=supervisor may launch a managed run")
    repo_root = Path(repo_root).resolve()
    if not command:
        raise HarnessError("A launch command is required")
    if not mock:
        _validate_production_command(command, repo_root)

    gpu = gpu_snapshot()
    disk = disk_snapshot(repo_root)
    if require_rtx4090:
        names = [entry["name"] for entry in gpu.get("gpus", [])]
        if not any("RTX 4090" in name for name in names):
            raise HarnessError("RTX 4090 preflight failed; detected GPUs: {}".format(names))
        if disk["free_gib"] < 10.0:
            raise HarnessError(
                "Managed training requires at least 10 GiB free; found {:.2f} GiB".format(
                    disk["free_gib"]
                )
            )

    phase_slug = _slug(phase, "phase")
    experiment_slug = _slug(experiment, "experiment")
    label_slug = _slug(label, "run")
    timestamp = dt.datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
    run_id = "{}_{}_{}".format(label_slug, timestamp, uuid.uuid4().hex[:6])
    root = Path(output_root).resolve() if output_root else repo_root / "logs" / "managed"
    run_dir = root / phase_slug / experiment_slug / run_id
    command = _materialize_managed_identity(command, run_dir, run_id)
    if not mock:
        command = _inject_managed_identity(command, run_dir, run_id)

    store = RunStore(run_dir)
    manifest = {
        "schema_version": 1,
        "run_id": run_id,
        "parent_run_id": parent_run_id,
        "phase": phase_slug,
        "experiment": experiment_slug,
        "label": label_slug,
        "authority": authority,
        "hardware_policy": "RTX 4090 only",
        "morai_reported_running": bool(morai_reported_running),
        "created_at": _now_iso(),
        "repo_root": str(repo_root),
        "run_dir": str(run_dir),
        "command": command,
        "git": _repo_snapshot(repo_root),
        "gpu_preflight": gpu,
        "disk_preflight": disk,
        "mock": bool(mock),
        "trend_min_iteration": max(int(trend_min_iteration), 0),
    }
    store.initialize(manifest)

    supervisor_log = (run_dir / "supervisor.log").open("ab", buffering=0)
    supervisor_command = [
        sys.executable,
        "-m",
        "legged_gym.harness.cli",
        "_supervise",
        "--run-dir",
        str(run_dir),
    ]
    try:
        process = subprocess.Popen(
            supervisor_command,
            cwd=str(repo_root),
            stdin=subprocess.DEVNULL,
            stdout=supervisor_log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            close_fds=True,
        )
    finally:
        supervisor_log.close()
    start_ticks = _proc_start_ticks(process.pid)
    _LOCAL_SUPERVISOR_HANDLES[process.pid] = process
    store.update_manifest(
        supervisor_pid=process.pid,
        supervisor_pgid=os.getpgid(process.pid),
        supervisor_start_ticks=start_ticks,
        supervisor_sid=os.getsid(process.pid),
        boot_id=_read_boot_id(),
        launched_at=_now_iso(),
    )
    store.transition(
        RunState.RUNNING,
        reason="supervisor launch",
        expected_states={RunState.PREFLIGHT.value},
    )
    return {"run_id": run_id, "run_dir": str(run_dir), "supervisor_pid": process.pid}


def supervise(run_dir):
    """Internal child process: own command lifetime and persist its exit status."""
    store = RunStore(run_dir)
    manifest = store.manifest()
    command = manifest["command"]
    received = {"signal": None}

    def remember_signal(signum, _frame):
        received["signal"] = int(signum)

    signal.signal(signal.SIGINT, remember_signal)
    signal.signal(signal.SIGTERM, remember_signal)
    console_path = store.run_dir / "console.log"
    with console_path.open("ab", buffering=0) as console:
        child = subprocess.Popen(
            command,
            cwd=manifest["repo_root"],
            stdin=subprocess.DEVNULL,
            stdout=console,
            stderr=subprocess.STDOUT,
            close_fds=True,
        )
        _atomic_write_json(
            store.run_dir / "process.json",
            {"train_pid": child.pid, "started_at": _now_iso(), "command": command},
        )
        supervisor_pid = os.getpid()
        supervisor_start_ticks = _proc_start_ticks(supervisor_pid)

        def write_heartbeat(child_return_code):
            _atomic_write_json(
                store.run_dir / "supervisor_heartbeat.json",
                {
                    "schema_version": 1,
                    "run_id": manifest["run_id"],
                    "supervisor_pid": supervisor_pid,
                    "supervisor_start_ticks": supervisor_start_ticks,
                    "train_pid": child.pid,
                    "updated_at": _now_iso(),
                    "wall_time_unix": time.time(),
                    "child_return_code": child_return_code,
                },
            )

        write_heartbeat(None)
        while True:
            return_code = child.poll()
            if return_code is not None:
                break
            time.sleep(_HEARTBEAT_INTERVAL_SECONDS)
            write_heartbeat(None)
        write_heartbeat(return_code)

    exit_record = {
        "return_code": return_code,
        "received_signal": received["signal"],
        "finished_at": _now_iso(),
    }
    _atomic_write_json(store.run_dir / "exit_status.json", exit_record)
    state = store.state()
    if state["state"] == RunState.STOPPING.value:
        kind = (state.get("stop") or {}).get("kind", "operator")
        if kind == "trend":
            target = RunState.STOPPED_TREND
        elif kind == "hard":
            target = RunState.ERROR
        else:
            target = RunState.STOPPED_OPERATOR
        store.transition(
            target,
            reason=(state.get("stop") or {}).get("reason"),
            expected_states={RunState.STOPPING.value},
            exit=exit_record,
        )
    elif state["state"] in {RunState.RUNNING.value, RunState.PREFLIGHT.value} and return_code == 0:
        completion_error = _completion_artifact_error(store.run_dir, bool(manifest.get("mock")))
        if completion_error is None:
            store.transition(
                RunState.COMPLETED,
                reason="process exited successfully",
                expected_states={RunState.RUNNING.value, RunState.PREFLIGHT.value},
                exit=exit_record,
            )
        else:
            store.transition(
                RunState.ERROR,
                reason=completion_error,
                expected_states={RunState.RUNNING.value, RunState.PREFLIGHT.value},
                exit=exit_record,
                last_error={"kind": "completion_artifact", "detail": completion_error},
            )
    elif state["state"] in {RunState.RUNNING.value, RunState.PREFLIGHT.value}:
        store.transition(
            RunState.ERROR,
            reason="process exited with code {}".format(return_code),
            expected_states={RunState.RUNNING.value, RunState.PREFLIGHT.value},
            exit=exit_record,
            last_error={"kind": "process_exit", "return_code": return_code},
        )
    return return_code


def _read_metrics(run_dir, limit=None):
    path = Path(run_dir) / "metrics.jsonl"
    if not path.exists():
        return []
    records = []
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict):
                records.append(value)
    return records[-limit:] if limit else records


def _completion_artifact_error(run_dir, mock):
    run_dir = Path(run_dir)
    metrics_path = run_dir / "metrics.jsonl"
    if not metrics_path.is_file() or metrics_path.stat().st_size == 0:
        return "successful process exit without non-empty metrics.jsonl"
    if not _read_metrics(run_dir, limit=1):
        return "successful process exit without a valid metrics record"
    if mock:
        return None
    if not (run_dir / "resolved_config.json").is_file():
        return "successful training exit without resolved_config.json"
    checkpoints = [
        path
        for path in run_dir.iterdir()
        if re.fullmatch(r"model_\d+\.pt", path.name) and path.stat().st_size > 0
    ]
    if not checkpoints:
        return "successful training exit without a stable checkpoint"
    if any(".tmp-" in path.name for path in run_dir.iterdir()):
        return "temporary checkpoint remained after successful exit"
    return None


def _validate_teacher45_p0_gate(run_dir):
    """Validate the exact two-iteration production P0 artifact contract."""
    run_dir = Path(run_dir).resolve()
    failures = []
    warnings = []

    def fail(code, message, **details):
        failure = {"code": code, "message": message}
        failure.update(details)
        failures.append(failure)

    try:
        manifest = _read_json(run_dir / "manifest.json")
    except (OSError, json.JSONDecodeError) as exc:
        manifest = None
        fail("manifest_invalid", "manifest.json is missing or invalid", error=str(exc))
    if not isinstance(manifest, dict):
        if not any(item["code"] == "manifest_invalid" for item in failures):
            fail("manifest_invalid", "manifest.json is missing or invalid")
        manifest = {}

    expected_manifest = {
        "phase": "p00-harness",
        "mock": False,
        "authority": "supervisor",
    }
    for key, expected in expected_manifest.items():
        actual = manifest.get(key)
        if actual != expected:
            fail(
                "manifest_{}_mismatch".format(key),
                "manifest {} must be {!r}".format(key, expected),
                expected=expected,
                actual=actual,
            )

    gpu_names = [
        entry.get("name", "")
        for entry in (manifest.get("gpu_preflight") or {}).get("gpus", [])
        if isinstance(entry, dict)
    ]
    if not any("RTX 4090" in name for name in gpu_names):
        fail(
            "gpu_preflight_mismatch",
            "launch preflight must name an RTX 4090",
            detected_gpu_names=gpu_names,
        )

    try:
        exit_record = _read_json(run_dir / "exit_status.json")
    except (OSError, json.JSONDecodeError) as exc:
        exit_record = None
        fail("exit_status_invalid", "exit_status.json is missing or invalid", error=str(exc))
    if not isinstance(exit_record, dict):
        if not any(item["code"] == "exit_status_invalid" for item in failures):
            fail("exit_status_invalid", "exit_status.json is missing or invalid")
        exit_record = {}
    if exit_record.get("return_code") != 0:
        fail("exit_code_nonzero", "training process exit code must be zero", actual=exit_record.get("return_code"))
    if exit_record.get("received_signal") is not None:
        fail(
            "exit_signal_present",
            "training process must not have received a signal",
            actual=exit_record.get("received_signal"),
        )

    console_path = run_dir / "console.log"
    console = ""
    if console_path.is_file() and console_path.stat().st_size > 0:
        console = console_path.read_text(encoding="utf-8", errors="replace")
    else:
        fail("console_missing", "console.log must exist and be nonempty")
    if console:
        if "Using GPU PhysX" not in console or "GPU Pipeline: enabled" not in console:
            fail("console_gpu_evidence_missing", "console.log must record GPU PhysX and the enabled GPU pipeline")
        for pattern in _HARD_ERROR_PATTERNS:
            match = pattern.search(console)
            if match:
                fail(
                    "console_hard_error",
                    "console.log contains a hard-error signature",
                    match=match.group(0),
                )
                break

    try:
        resolved = _read_json(run_dir / "resolved_config.json")
    except (OSError, json.JSONDecodeError) as exc:
        resolved = None
        fail("resolved_config_invalid", "resolved_config.json is missing or invalid", error=str(exc))
    if not isinstance(resolved, dict):
        if not any(item["code"] == "resolved_config_invalid" for item in failures):
            fail("resolved_config_invalid", "resolved_config.json is missing or invalid")
        resolved = {}

    config_contract = (
        (("task",), "a1_limping_base_v2"),
        (("environment", "env", "task_name"), "a1_limping_base_v2"),
        (("environment", "env", "num_envs"), 64),
        (("environment", "env", "num_observations"), 45),
        (("environment", "env", "num_privileged_obs"), 45),
        (("environment", "terrain", "mesh_type"), "plane"),
        (("environment", "terrain", "measure_heights"), False),
        (("environment", "rewards", "only_positive_rewards"), True),
        (("environment", "seed"), 1),
        (("schema", "actor_observation_dim"), 45),
        (("schema", "privileged_observation_dim"), 45),
        (("schema", "teacher_latent_dim"), 8),
        (("schema", "actor_input_dim"), 53),
        (("schema", "critic_input_dim"), 53),
        (("training", "seed"), 1),
        (("training", "runner", "num_steps_per_env"), 24),
        (("training", "algorithm", "num_learning_epochs"), 5),
        (("training", "algorithm", "num_mini_batches"), 4),
        (("training", "algorithm", "entropy_coef"), 0.01),
        (("training", "algorithm", "desired_kl"), 0.01),
        (("training", "runner", "max_iterations"), 2),
        (("training", "runner", "save_interval"), 1),
    )
    for path, expected in config_contract:
        actual = resolved
        for part in path:
            actual = actual.get(part) if isinstance(actual, dict) else None
        if actual != expected:
            dotted = ".".join(path)
            fail(
                "config_mismatch",
                "resolved configuration {} must be {!r}".format(dotted, expected),
                field=dotted,
                expected=expected,
                actual=actual,
            )
    algorithm_config = ((resolved.get("training") or {}).get("algorithm") or {})
    if "max_policy_kl" in algorithm_config:
        fail(
            "config_local_hard_kl_present",
            "resolved configuration must not contain local max_policy_kl",
            actual=algorithm_config.get("max_policy_kl"),
        )

    metrics_path = run_dir / "metrics.jsonl"
    records = []
    invalid_metric_lines = []
    if metrics_path.is_file() and metrics_path.stat().st_size > 0:
        with metrics_path.open("r", encoding="utf-8", errors="replace") as handle:
            for line_number, line in enumerate(handle, 1):
                try:
                    value = json.loads(line)
                except json.JSONDecodeError:
                    invalid_metric_lines.append(line_number)
                    continue
                if isinstance(value, dict):
                    records.append(value)
                else:
                    invalid_metric_lines.append(line_number)
    else:
        fail("metrics_missing", "metrics.jsonl must exist and be nonempty")
    if invalid_metric_lines:
        fail(
            "metrics_malformed",
            "every metrics.jsonl line must be a JSON object",
            line_numbers=invalid_metric_lines,
        )
    iterations = [record.get("iteration") for record in records]
    transitions = [record.get("total_transitions") for record in records]
    if iterations != [0, 1]:
        fail("metric_iterations_mismatch", "metric iterations must be exactly [0, 1]", actual=iterations)
    if transitions != [1536, 3072]:
        fail(
            "metric_transitions_mismatch",
            "metric transition totals must be exactly [1536, 3072]",
            actual=transitions,
        )

    expected_ppo_updates = 5 * 4
    for index, record in enumerate(records):
        nonfinite = sorted(
            key
            for key, value in record.items()
            if isinstance(value, (int, float))
            and not isinstance(value, bool)
            and not math.isfinite(float(value))
        )
        if nonfinite:
            fail(
                "metric_nonfinite",
                "all numeric metrics must be finite",
                record_index=index,
                metrics=nonfinite,
            )
        completed = record.get("PPO/completed_updates")
        planned = record.get("PPO/planned_updates")
        if (
            completed != expected_ppo_updates
            or planned != expected_ppo_updates
            or completed != planned
        ):
            fail(
                "ppo_updates_mismatch",
                "completed and planned PPO updates must both equal {}".format(
                    expected_ppo_updates
                ),
                record_index=index,
                completed_updates=completed,
                planned_updates=planned,
            )
        skipped = record.get("PPO/nonfinite_update_skipped")
        if skipped != 0:
            fail(
                "ppo_nonfinite_update_skipped",
                "PPO/nonfinite_update_skipped must be zero",
                record_index=index,
                actual=skipped,
            )
        if record.get("PPO/hard_kl_stopped") != 0:
            fail(
                "ppo_hard_kl_stop_present",
                "PPO/hard_kl_stopped must be zero",
                record_index=index,
                actual=record.get("PPO/hard_kl_stopped"),
            )
        for metric_name, failure_code in (
            ("PPO/mean_kl", "ppo_mean_kl_invalid"),
            ("PPO/max_kl", "ppo_max_kl_invalid"),
        ):
            value = record.get(metric_name)
            if (
                not isinstance(value, (int, float))
                or isinstance(value, bool)
                or not math.isfinite(float(value))
                or float(value) < 0.0
            ):
                fail(
                    failure_code,
                    "{} must be numeric, finite, and nonnegative".format(metric_name),
                    record_index=index,
                    actual=value,
                )

    inventory = {item["iteration"]: item for item in _checkpoint_inventory(run_dir)}
    missing_checkpoints = [iteration for iteration in (0, 1, 2) if iteration not in inventory]
    if missing_checkpoints:
        fail(
            "checkpoints_missing",
            "nonempty model_0.pt, model_1.pt, and model_2.pt are required",
            missing_iterations=missing_checkpoints,
        )
    temporary_checkpoints = sorted(
        path.name for path in run_dir.iterdir() if path.is_file() and ".tmp-" in path.name
    ) if run_dir.is_dir() else []
    if temporary_checkpoints:
        fail(
            "temporary_checkpoint_present",
            "temporary checkpoint files must not remain",
            files=temporary_checkpoints,
        )

    try:
        latest_snapshot = _read_json(run_dir / "latest_snapshot.json")
    except (OSError, json.JSONDecodeError):
        latest_snapshot = None
    if isinstance(latest_snapshot, dict):
        post_run_gpu = latest_snapshot.get("gpu")
        if isinstance(post_run_gpu, dict) and not post_run_gpu.get("available", False):
            warnings.append(
                {
                    "code": "post_run_gpu_telemetry_unavailable",
                    "message": "post-run GPU telemetry is unavailable; launch preflight remains authoritative",
                    "error": post_run_gpu.get("error"),
                }
            )

    return {
        "run_dir": str(run_dir),
        "valid": not failures,
        "failures": failures,
        "warnings": warnings,
        "summary": {
            "failure_count": len(failures),
            "warning_count": len(warnings),
            "metric_records": len(records),
            "checkpoint_iterations": sorted(inventory),
        },
    }


def _nested(value, *parts):
    for part in parts:
        value = value.get(part) if isinstance(value, dict) else None
    return value


def _validate_official_wim_p0_gate(run_dir, profile):
    run_dir = Path(run_dir).resolve()
    failures = []

    def fail(code, message, **details):
        item = {"code": code, "message": message}
        item.update(details)
        failures.append(item)

    try:
        manifest = _read_json(run_dir / "manifest.json")
        resolved = _read_json(run_dir / "resolved_config.json")
        exit_record = _read_json(run_dir / "exit_status.json")
    except (OSError, json.JSONDecodeError) as exc:
        manifest, resolved, exit_record = {}, {}, {}
        fail("core_artifact_invalid", "manifest/resolved/exit artifact missing or invalid", error=str(exc))
    manifest = manifest if isinstance(manifest, dict) else {}
    resolved = resolved if isinstance(resolved, dict) else {}
    exit_record = exit_record if isinstance(exit_record, dict) else {}
    if manifest.get("mock") is not False or manifest.get("authority") != "supervisor":
        fail("manifest_contract", "P0 must be a non-mock supervisor run")
    if exit_record.get("return_code") != 0 or exit_record.get("received_signal") is not None:
        fail("exit_contract", "P0 must exit normally", actual=exit_record)

    checks = (
        (("task",), profile["task"]),
        (("environment", "env", "num_envs"), profile["num_envs"]),
        (("environment", "env", "num_observations"), profile["num_observations"]),
        (("environment", "env", "num_privileged_obs"), profile["num_privileged_obs"]),
        (("environment", "env", "num_actions"), profile["num_actions"]),
        (("environment", "terrain", "mesh_type"), profile["mesh_type"]),
        (("environment", "terrain", "measure_heights"), profile["measure_heights"]),
        (("environment", "terrain", "curriculum"), True),
        (("environment", "control", "control_type"), "P"),
        (("environment", "control", "stiffness"), {"joint": 20.0}),
        (("environment", "control", "damping"), {"joint": 0.5}),
        (("environment", "control", "action_scale"), 0.25),
        (("environment", "normalization", "clip_actions"), 100.0),
        (("environment", "rewards", "only_positive_rewards"), True),
        (("environment", "asset", "terminate_after_contacts_on"), ["base"]),
        (("environment", "domain_rand", "randomize_friction"), True),
        (("seed",), profile["seed"]),
        (("training", "runner", "num_steps_per_env"), profile["num_steps_per_env"]),
        (("training", "runner", "max_iterations"), profile["max_iterations"]),
        (("training", "runner", "save_interval"), profile["save_interval"]),
        (("training", "algorithm", "num_mini_batches"), profile["num_mini_batches"]),
        (("training", "algorithm", "num_learning_epochs"), profile["num_learning_epochs"]),
        (("resume",), False),
        (("no_resume", "load_run"), None),
        (("no_resume", "checkpoint"), None),
    )
    for path, expected in checks:
        actual = _nested(resolved, *path)
        if actual != expected:
            fail("config_mismatch", "{} must equal {!r}".format(".".join(path), expected), field=".".join(path), expected=expected, actual=actual)
    for axis, expected in (("measured_points_x", 17), ("measured_points_y", 11)):
        actual = _nested(resolved, "environment", "terrain", axis)
        if not isinstance(actual, list) or len(actual) != expected:
            fail("height_grid_mismatch", "{} must contain exactly {} points".format(axis, expected), actual=actual)
    if _nested(resolved, "official_sources", "legged_gym", "forbidden_diffs"):
        fail("forbidden_source_diff", "official source conformance contains forbidden differences")
    if not _nested(resolved, "official_sources", "rsl_rl", "pass"):
        fail("rsl_rl_conformance", "official rsl_rl revision/hash/clean contract failed")

    def read_rows(name):
        rows = []
        path = run_dir / name
        try:
            for line in path.read_text(encoding="utf-8").splitlines():
                rows.append(json.loads(line))
        except (OSError, json.JSONDecodeError) as exc:
            fail(name + "_invalid", name + " is missing or malformed", error=str(exc))
        return rows

    rows = read_rows("metrics.jsonl")
    mirror = read_rows("wandb_metrics.jsonl")
    if [row.get("iteration") for row in rows] != list(profile["metric_iterations"]):
        fail("metric_iterations", "metric iterations must be exactly [0, 1]")
    if rows != mirror:
        fail("wandb_mirror_mismatch", "W&B mirror rows must canonically equal local metrics")
    for index, row in enumerate(rows):
        if row.get("finite") is not True or any(
            not math.isfinite(float(value)) for value in row.values()
            if isinstance(value, (int, float)) and not isinstance(value, bool)
        ):
            fail("metric_nonfinite", "all metric rows must be explicitly finite", record_index=index)

    inventory = _checkpoint_inventory(run_dir)
    iterations = [item["iteration"] for item in inventory]
    if iterations != list(profile["checkpoint_iterations"]):
        fail("checkpoint_inventory", "checkpoint inventory must be exactly model_0/1/2.pt", actual=iterations)
    unexpected = sorted(path.name for path in run_dir.glob("model_*.pt") if not re.fullmatch(r"model_[012]\.pt", path.name))
    temporary = sorted(path.name for path in run_dir.iterdir() if ".tmp-" in path.name)
    if unexpected or temporary:
        fail("checkpoint_artifacts", "unexpected or temporary checkpoints found", unexpected=unexpected, temporary=temporary)

    hashes = {}
    for name in ("manifest.json", "resolved_config.json", "metrics.jsonl", "wandb_metrics.jsonl"):
        path = run_dir / name
        if path.is_file():
            hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    hashes.update(
        {
            Path(item["path"]).name: hashlib.sha256(Path(item["path"]).read_bytes()).hexdigest()
            for item in inventory
        }
    )
    report = {"run_dir": str(run_dir), "profile": profile["name"], "valid": not failures, "failures": failures, "artifact_sha256": hashes}
    report_path = run_dir / "p0_validation_report.json"
    _atomic_write_json(report_path, report)
    return report


def validate_p0_gate(run_dir, profile="teacher45"):
    """Validate a named immutable P0 contract; teacher45 remains unchanged."""
    from legged_gym.harness.p0_profiles import P0_PROFILES
    if profile not in P0_PROFILES:
        raise HarnessError("Unknown P0 profile: {}".format(profile))
    selected = P0_PROFILES[profile]
    if selected.get("validator") == "legacy":
        return _validate_teacher45_p0_gate(run_dir)
    return _validate_official_wim_p0_gate(run_dir, selected)


def _console_tail(run_dir, max_bytes=32768):
    path = Path(run_dir) / "console.log"
    if not path.exists():
        return ""
    with path.open("rb") as handle:
        handle.seek(0, os.SEEK_END)
        size = handle.tell()
        handle.seek(max(0, size - max_bytes))
        return handle.read().decode("utf-8", errors="replace")


def _metric(records, candidates):
    values = []
    for record in records:
        for key in candidates:
            value = record.get(key)
            if isinstance(value, (int, float)) and math.isfinite(float(value)):
                values.append(float(value))
                break
    return values


def _valid_survival_metric(records):
    values = []
    for record in records:
        if "Episode/survival_rate_20s" in record:
            valid = record.get("Episode/survival_rate_20s_valid", 0.0)
            value = record.get("Episode/survival_rate_20s")
            if float(valid or 0.0) >= 0.5 and isinstance(value, (int, float)) and math.isfinite(float(value)):
                values.append(float(value))
        elif "Rollout/survival_rate_20s" in record:
            value = record.get("Rollout/survival_rate_20s")
            if isinstance(value, (int, float)) and math.isfinite(float(value)):
                values.append(float(value))
    return values


def _checkpoint_inventory(run_dir):
    run_dir = Path(run_dir)
    checkpoints = []
    if not run_dir.exists():
        return checkpoints
    for path in run_dir.iterdir():
        match = re.fullmatch(r"model_(\d+)\.pt", path.name)
        if match is not None and path.is_file() and path.stat().st_size > 0:
            checkpoints.append(
                {"iteration": int(match.group(1)), "path": str(path), "size": path.stat().st_size}
            )
    return sorted(checkpoints, key=lambda item: item["iteration"])


def _relative_improvement(values, lower_is_better=True):
    if len(values) < 4:
        return None
    midpoint = len(values) // 2
    old = statistics.median(values[:midpoint])
    new = statistics.median(values[midpoint:])
    denominator = max(abs(old), 1.0e-8)
    return (old - new) / denominator if lower_is_better else (new - old) / denominator


def assess_trend(records, min_iteration=250, window_points=12, checkpoint_available=True):
    """Return flexible evidence for tracker reasoning; never stop a run itself."""
    records = records[-max(int(window_points), 6):]
    latest_iteration = int(records[-1].get("iteration", -1)) if records else -1
    result = {
        "eligible": (
            latest_iteration >= int(min_iteration)
            and len(records) >= 6
            and bool(checkpoint_available)
        ),
        "candidate_stop": False,
        "confidence": 0.0,
        "latest_iteration": latest_iteration,
        "reasons": [],
        "evidence": {},
    }
    if not result["eligible"]:
        if not checkpoint_available:
            result["reasons"].append("no stable checkpoint is available")
        else:
            result["reasons"].append("insufficient burn-in or metric history")
        return result

    vx = _metric(records, ("Rollout/command_vx_rmse", "Episode/command_vx_rmse"))
    vertical = _metric(
        records,
        (
            "Rollout/vertical_velocity_rmse",
            "Rollout/vertical_velocity_rms",
            "Episode/vertical_velocity_rms",
        ),
    )
    four_feet = _metric(records, ("Rollout/four_feet_contact_rate", "Episode/four_feet_contact_rate"))
    survival = _valid_survival_metric(records)
    evidence = {
        "vx_rmse_latest": vx[-1] if vx else None,
        "vx_rmse_improvement": _relative_improvement(vx),
        "vertical_rms_latest": vertical[-1] if vertical else None,
        "vertical_rms_improvement": _relative_improvement(vertical),
        "four_feet_latest": four_feet[-1] if four_feet else None,
        "four_feet_improvement": _relative_improvement(four_feet),
        "survival_latest": survival[-1] if survival else None,
        "survival_improvement": _relative_improvement(survival, lower_is_better=False),
    }
    result["evidence"] = evidence

    tracking_bad = (
        evidence["vx_rmse_latest"] is not None
        and evidence["vx_rmse_latest"] > 0.25
        and (evidence["vx_rmse_improvement"] or 0.0) < 0.05
    )
    bounce_bad = (
        evidence["vertical_rms_latest"] is not None
        and evidence["vertical_rms_latest"] > 0.35
        and (evidence["vertical_rms_improvement"] or 0.0) < 0.05
    )
    contact_bad = (
        evidence["four_feet_latest"] is not None
        and evidence["four_feet_latest"] > 0.60
        and (evidence["four_feet_improvement"] or 0.0) < 0.05
    )
    survival_bad = evidence["survival_latest"] is not None and evidence["survival_latest"] < 0.80

    if tracking_bad:
        result["reasons"].append("command tracking is poor and stalled")
    if bounce_bad:
        result["reasons"].append("vertical bounce is high and not improving")
    if contact_bad:
        result["reasons"].append("four-feet contact remains high and not improving")
    if survival_bad:
        result["reasons"].append("20-second survival is low")
    corroborating = sum((bounce_bad, contact_bad, survival_bad))
    result["candidate_stop"] = bool(tracking_bad and corroborating >= 1)
    if result["candidate_stop"]:
        result["confidence"] = min(0.95, 0.55 + 0.12 * corroborating)
    return result


def collect_run(run_dir, stale_seconds=600, trend_min_iteration=250):
    """Collect one durable snapshot without taking a trend-stop decision."""
    store = RunStore(run_dir)
    manifest = store.manifest()
    state = store.state()
    pid = manifest.get("supervisor_pid")
    process_visible = bool(pid) and _process_alive(pid, manifest.get("supervisor_start_ticks"))
    exit_record = _read_json(store.run_dir / "exit_status.json")
    heartbeat = _read_json(store.run_dir / "supervisor_heartbeat.json")
    heartbeat_age = _supervisor_heartbeat(manifest, heartbeat)
    heartbeat_fresh = heartbeat_age is not None and heartbeat_age <= _HEARTBEAT_FRESH_SECONDS
    alive = bool(process_visible or heartbeat_fresh)
    alerts = []

    if exit_record or (not process_visible and not heartbeat_fresh):
        local_handle = _LOCAL_SUPERVISOR_HANDLES.pop(int(pid or 0), None)
        if local_handle is not None:
            try:
                local_handle.wait(timeout=1)
            except subprocess.TimeoutExpired:
                _LOCAL_SUPERVISOR_HANDLES[int(pid)] = local_handle

    if state["state"] in _ACTIVE_STATES and exit_record:
        if state["state"] == RunState.STOPPING.value:
            kind = (state.get("stop") or {}).get("kind", "operator")
            target = RunState.STOPPED_TREND if kind == "trend" else RunState.STOPPED_OPERATOR
            if kind == "hard":
                target = RunState.ERROR
            state = store.transition(
                target,
                reason="supervisor exited after stop request",
                expected_states={RunState.STOPPING.value},
                exit=exit_record,
            )
        elif exit_record.get("return_code") == 0:
            completion_error = _completion_artifact_error(
                store.run_dir, bool(manifest.get("mock"))
            )
            if completion_error is None:
                state = store.transition(
                    RunState.COMPLETED,
                    reason="supervisor recorded successful exit",
                    expected_states=_ACTIVE_STATES,
                    exit=exit_record,
                )
            else:
                state = store.transition(
                    RunState.ERROR,
                    reason=completion_error,
                    expected_states=_ACTIVE_STATES,
                    exit=exit_record,
                    last_error={"kind": "completion_artifact", "detail": completion_error},
                )
        else:
            state = store.transition(
                RunState.ERROR,
                reason="process exited with code {}".format(exit_record.get("return_code")),
                expected_states=_ACTIVE_STATES,
                exit=exit_record,
                last_error={"kind": "process_exit", "exit": exit_record},
            )
        _clear_liveness_probe(store.run_dir)
    elif state["state"] in _ACTIVE_STATES and alive:
        _clear_liveness_probe(store.run_dir)
        if not process_visible:
            alerts.append(
                {
                    "severity": "warning",
                    "kind": "supervisor_pid_unobservable",
                    "heartbeat_age_seconds": heartbeat_age,
                }
            )
    elif state["state"] in _ACTIVE_STATES:
        probe_path = store.run_dir / "liveness_probe.json"
        now = time.time()
        probe = _read_json(probe_path, default={})
        first_missing_unix = probe.get("first_missing_wall_time_unix")
        try:
            elapsed = max(0.0, now - float(first_missing_unix))
        except (TypeError, ValueError):
            elapsed = 0.0
            probe = {
                "schema_version": 1,
                "run_id": manifest["run_id"],
                "first_missing_at": _now_iso(),
                "first_missing_wall_time_unix": now,
                "consecutive_misses": 0,
            }
        probe.update(
            {
                "last_missing_at": _now_iso(),
                "last_missing_wall_time_unix": now,
                "consecutive_misses": int(probe.get("consecutive_misses", 0)) + 1,
                "heartbeat_age_seconds": heartbeat_age,
                "progress_signature": _progress_signature(store.run_dir),
            }
        )
        _atomic_write_json(probe_path, probe)
        alerts.append(
            {
                "severity": "warning",
                "kind": "supervisor_pid_unobservable",
                "heartbeat_age_seconds": heartbeat_age,
                "consecutive_misses": probe["consecutive_misses"],
                "grace_elapsed_seconds": elapsed,
            }
        )
        if probe["consecutive_misses"] >= 2 and elapsed >= _LIVENESS_GRACE_SECONDS:
            state = store.transition(
                RunState.ERROR,
                reason="supervisor heartbeat remained stale after repeated grace-window misses",
                expected_states=_ACTIVE_STATES,
                exit=None,
                last_error={"kind": "missing_supervisor", "liveness_probe": probe},
            )

    records = _read_metrics(run_dir, limit=120)
    latest = records[-1] if records else None
    if latest:
        for key, value in latest.items():
            if isinstance(value, (int, float)) and not math.isfinite(float(value)):
                alerts.append({"severity": "hard", "kind": "nonfinite_metric", "metric": key})
        if float(latest.get("PPO/nonfinite_update_skipped", 0.0) or 0.0) > 0:
            alerts.append({"severity": "hard", "kind": "ppo_nonfinite_update"})

    console_tail = _console_tail(run_dir)
    if state["state"] == RunState.ERROR.value:
        for pattern in _HARD_ERROR_PATTERNS:
            match = pattern.search(console_tail)
            if match:
                alerts.append({"severity": "hard", "kind": "console_error", "match": match.group(0)})
                break

    freshness_candidates = [store.run_dir / "metrics.jsonl", store.run_dir / "console.log"]
    mtimes = [path.stat().st_mtime for path in freshness_candidates if path.exists()]
    stale_for = max(0.0, time.time() - max(mtimes)) if mtimes else None
    if alive and stale_for is not None and stale_for > float(stale_seconds):
        alerts.append({"severity": "hard", "kind": "stale_output", "seconds": stale_for})

    checkpoints = _checkpoint_inventory(store.run_dir)
    snapshot = {
        "collected_at": _now_iso(),
        "run_id": manifest["run_id"],
        "run_dir": str(store.run_dir),
        "state": state,
        "process_alive": alive,
        "process_visible": process_visible,
        "supervisor_heartbeat": heartbeat,
        "heartbeat_age_seconds": heartbeat_age,
        "latest_metrics": latest,
        "metric_records": len(records),
        "stale_for_seconds": stale_for,
        "alerts": alerts,
        "trend": assess_trend(
            records,
            min_iteration=trend_min_iteration,
            checkpoint_available=bool(checkpoints),
        ),
        "checkpoints": checkpoints,
        "gpu": gpu_snapshot(),
        "exit": exit_record,
    }
    _atomic_write_json(store.run_dir / "latest_snapshot.json", snapshot)
    return snapshot


def _validate_supervisor_identity(store, manifest):
    pid = int(manifest.get("supervisor_pid") or 0)
    pgid = int(manifest.get("supervisor_pgid") or 0)
    if pid <= 1 or pgid <= 1:
        raise HarnessError("Refusing stop: invalid supervisor pid/pgid")
    expected_boot_id = manifest.get("boot_id")
    current_boot_id = _read_boot_id()
    if expected_boot_id and current_boot_id != expected_boot_id:
        raise HarnessError("Refusing stop: host boot identity changed")
    if not _process_alive(pid, manifest.get("supervisor_start_ticks")):
        heartbeat = _read_json(store.run_dir / "supervisor_heartbeat.json")
        heartbeat_age = _supervisor_heartbeat(manifest, heartbeat)
        if heartbeat_age is not None and heartbeat_age <= _HEARTBEAT_FRESH_SECONDS:
            raise HarnessError(
                "Refusing stop: execution-domain mismatch; fresh heartbeat exists but "
                "supervisor PID identity is not visible"
            )
        raise HarnessError(
            "Refusing stop: supervisor PID identity is unavailable in this execution domain"
        )
    try:
        current_pgid = os.getpgid(pid)
        cmdline = Path("/proc/{}/cmdline".format(pid)).read_bytes().replace(b"\x00", b" ").decode()
    except OSError as exc:
        raise HarnessError("Cannot validate supervisor identity: {}".format(exc))
    if current_pgid != pgid:
        raise HarnessError("Refusing stop: process-group identity changed")
    if pgid == os.getpgrp():
        raise HarnessError("Refusing stop: supervisor shares the caller's process group")
    if "legged_gym.harness.cli" not in cmdline or "_supervise" not in cmdline:
        raise HarnessError("Refusing stop: PID is not the managed supervisor")
    if str(store.run_dir) not in cmdline:
        raise HarnessError("Refusing stop: supervisor belongs to another run")
    return pid, pgid, True


def _validate_stop_request(store, manifest, authority, kind, decision):
    if authority not in {"supervisor", "tracker"}:
        raise HarnessError("Only authority=supervisor or tracker may stop a managed run")
    if kind == "operator" and authority != "supervisor":
        raise HarnessError("Only authority=supervisor may request an operator stop")
    if kind != "trend":
        return

    records = _read_metrics(store.run_dir, limit=120)
    latest_iteration = int(records[-1].get("iteration", -1)) if records else -1
    min_iteration = max(int(manifest.get("trend_min_iteration", 250)), 0)
    if latest_iteration < min_iteration:
        raise HarnessError(
            "Trend stop is ineligible before iteration {} (latest {})".format(
                min_iteration, latest_iteration
            )
        )
    if not _checkpoint_inventory(store.run_dir):
        raise HarnessError("Trend stop requires a preserved checkpoint")
    required = {
        "evidence_window",
        "alternatives_considered",
        "confidence",
        "continuation_cost",
        "last_usable_checkpoint",
    }
    missing = sorted(required.difference(decision or {}))
    if missing:
        raise HarnessError(
            "Trend stop decision is missing keys: {}".format(", ".join(missing))
        )


def stop_run(
    run_dir,
    kind,
    reason,
    authority,
    decision=None,
    grace_seconds=30.0,
):
    """Gracefully stop only the exact recorded process group."""
    if kind not in {"trend", "operator", "hard"}:
        raise HarnessError("Unknown stop kind: {}".format(kind))
    store = RunStore(run_dir)
    manifest = store.manifest()
    _validate_stop_request(store, manifest, authority, kind, decision)
    state = store.state()
    if state["state"] not in {RunState.RUNNING.value, RunState.PREFLIGHT.value}:
        raise HarnessError("Run is not stoppable from state {}".format(state["state"]))
    pid, pgid, alive = _validate_supervisor_identity(store, manifest)
    stop_record = {
        "authority": authority,
        "kind": kind,
        "reason": reason,
        "requested_at": _now_iso(),
        "decision": decision or {},
    }
    state = store.transition(
        RunState.STOPPING,
        reason=reason,
        expected_states={RunState.RUNNING.value, RunState.PREFLIGHT.value},
        stop=stop_record,
    )
    if state["state"] != RunState.STOPPING.value:
        return collect_run(run_dir)
    _atomic_write_json(store.run_dir / "tracker_decision.json", stop_record)
    if not alive:
        return collect_run(run_dir)

    os.killpg(pgid, signal.SIGINT)
    deadline = time.time() + max(float(grace_seconds), 0.1)
    while time.time() < deadline:
        if not _process_alive(pid, manifest.get("supervisor_start_ticks")):
            return collect_run(run_dir)
        time.sleep(0.1)
    os.killpg(pgid, signal.SIGTERM)
    second_deadline = time.time() + min(max(float(grace_seconds), 1.0), 10.0)
    while time.time() < second_deadline:
        if not _process_alive(pid, manifest.get("supervisor_start_ticks")):
            return collect_run(run_dir)
        time.sleep(0.1)
    os.killpg(pgid, signal.SIGKILL)
    forced_target = RunState.ERROR if kind == "hard" else (
        RunState.STOPPED_TREND if kind == "trend" else RunState.STOPPED_OPERATOR
    )
    store.transition(
        forced_target,
        reason="forced exact process-group stop after SIGINT/SIGTERM timeout",
        expected_states={RunState.STOPPING.value},
        forced_kill=True,
        last_error={"kind": "forced_process_kill", "pid": pid, "pgid": pgid},
    )
    return collect_run(run_dir)
