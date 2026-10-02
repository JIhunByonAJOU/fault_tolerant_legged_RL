"""Local ROS topic-rate evidence collector for managed training runs."""

import argparse
import datetime as dt
import json
import os
import threading
import time
import uuid
from pathlib import Path


def _now_iso():
    return dt.datetime.now().astimezone().isoformat(timespec="microseconds")


def _atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp-" + uuid.uuid4().hex)
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(str(temporary), str(path))


def _append_jsonl(path, value):
    with Path(path).open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _proc_start_ticks(pid):
    try:
        return int(Path("/proc/{}/stat".format(pid)).read_text(encoding="utf-8").split()[21])
    except (OSError, ValueError, IndexError):
        return None


class EgoTopicRateGuard:
    """Deterministic window evaluator; this class never sends a signal."""

    def __init__(self, run_dir, run_id, window_seconds=15.0, minimum_rate_hz=40.0):
        self.run_dir = Path(run_dir)
        self.run_id = str(run_id)
        self.window_seconds = float(window_seconds)
        self.minimum_rate_hz = float(minimum_rate_hz)
        self.guard_pid = os.getpid()
        self.guard_start_ticks = _proc_start_ticks(self.guard_pid)
        self.samples = []
        self.lock = threading.Lock()
        self.stage = "preflight"
        self.window_start_ns = None
        self.consecutive_below = 0
        self.window_sequence = 0
        self.subscriber_status = "starting"
        self.train_identity = {}
        self.decision = None

    def write_heartbeat(self, status=None):
        if status is not None:
            self.subscriber_status = status
        record = {
            "schema_version": 1,
            "run_id": self.run_id,
            "guard_pid": self.guard_pid,
            "guard_start_ticks": self.guard_start_ticks,
            "updated_at": _now_iso(),
            "wall_time_unix": time.time(),
            "monotonic_ns": time.monotonic_ns(),
            "subscriber_status": self.subscriber_status,
            "stage": self.stage,
            "decision": self.decision,
        }
        _atomic_json(self.run_dir / "resource_guard_state.json", record)
        return record

    def record_sample(self, monotonic_ns=None, wall_time_unix=None):
        monotonic_ns = time.monotonic_ns() if monotonic_ns is None else int(monotonic_ns)
        wall_time_unix = time.time() if wall_time_unix is None else float(wall_time_unix)
        record = {
            "schema_version": 1,
            "run_id": self.run_id,
            "stage": self.stage,
            "received_at": dt.datetime.fromtimestamp(wall_time_unix).astimezone().isoformat(
                timespec="microseconds"
            ),
            "wall_time_unix": wall_time_unix,
            "monotonic_ns": monotonic_ns,
        }
        with self.lock:
            self.samples.append(monotonic_ns)
        _append_jsonl(self.run_dir / "resource_samples.jsonl", record)
        return record

    def start_training_windows(self, start_ns, train_identity):
        self.stage = "running"
        self.window_start_ns = int(start_ns)
        self.train_identity = dict(train_identity)
        with self.lock:
            self.samples = [value for value in self.samples if value >= self.window_start_ns]
        self.write_heartbeat("subscribed")

    def close_window(self, end_ns, partial=False):
        if self.window_start_ns is None:
            raise ValueError("training window has not started")
        end_ns = int(end_ns)
        if end_ns < self.window_start_ns:
            raise ValueError("window end precedes start")
        start_ns = self.window_start_ns
        duration_s = (end_ns - start_ns) / 1e9
        with self.lock:
            arrivals = sorted(
                value for value in self.samples if start_ns <= value < end_ns
            )
            self.samples = [value for value in self.samples if value >= end_ns]
        boundaries = [start_ns] + arrivals + [end_ns]
        max_gap_s = max(
            ((right - left) / 1e9 for left, right in zip(boundaries, boundaries[1:])),
            default=duration_s,
        )
        rate_hz = len(arrivals) / duration_s if duration_s > 0 else 0.0
        end_wall = time.time()
        start_wall = end_wall - duration_s
        complete = not partial
        if complete and rate_hz < self.minimum_rate_hz:
            self.consecutive_below += 1
        elif complete:
            self.consecutive_below = 0
        record = {
            "schema_version": 1,
            "run_id": self.run_id,
            "sequence": self.window_sequence,
            "window_start_monotonic_ns": start_ns,
            "window_end_monotonic_ns": end_ns,
            "window_start": dt.datetime.fromtimestamp(start_wall).astimezone().isoformat(
                timespec="microseconds"
            ),
            "window_end": dt.datetime.fromtimestamp(end_wall).astimezone().isoformat(
                timespec="microseconds"
            ),
            "duration_s": duration_s,
            "message_count": len(arrivals),
            "rate_hz": rate_hz,
            "max_gap_s": max_gap_s,
            "complete": complete,
            "partial": bool(partial),
            "consecutive_below_40": self.consecutive_below,
            "subscriber_status": self.subscriber_status,
            "guard_pid": self.guard_pid,
            "guard_start_ticks": self.guard_start_ticks,
            "train_pid": self.train_identity.get("train_pid"),
            "train_start_ticks": self.train_identity.get("train_start_ticks"),
        }
        _append_jsonl(self.run_dir / "resource_windows.jsonl", record)
        self.window_sequence += 1
        self.window_start_ns = end_ns
        if complete and not arrivals:
            self.write_decision("no_message_window", record)
        elif complete and self.consecutive_below >= 2:
            self.write_decision("two_consecutive_below_rate", record)
        return record

    def write_decision(self, reason, evidence=None):
        if self.decision is not None:
            return self.decision
        self.decision = {
            "schema_version": 1,
            "run_id": self.run_id,
            "requested_at": _now_iso(),
            "wall_time_unix": time.time(),
            "kind": "hard",
            "reason": str(reason),
            "evidence": evidence or {},
            "guard_pid": self.guard_pid,
            "guard_start_ticks": self.guard_start_ticks,
            "train_pid": self.train_identity.get("train_pid"),
            "train_start_ticks": self.train_identity.get("train_start_ticks"),
        }
        _atomic_json(self.run_dir / "resource_guard_decision.json", self.decision)
        self.write_heartbeat("stop_requested")
        return self.decision

    def subscriber_exception(self, exc):
        self.subscriber_status = "exception"
        return self.write_decision("subscriber_exception", {"detail": str(exc)})

    def _ros_callback(self, _message):
        try:
            self.record_sample()
        except Exception as exc:
            self.subscriber_exception(exc)

    def run_preflight(self, topic, timeout_s):
        try:
            import rospy
        except Exception as exc:
            self.subscriber_exception(exc)
            return False
        try:
            rospy.init_node("managed_resource_guard", anonymous=True, disable_signals=True)
            rospy.get_master().getPid()
            published = {name for name, _type_name in rospy.get_published_topics()}
            if topic not in published:
                raise RuntimeError("ROS topic is not published: {}".format(topic))
            rospy.Subscriber(topic, rospy.AnyMsg, self._ros_callback, queue_size=2000)
            self.subscriber_status = "subscribed"
        except Exception as exc:
            self.subscriber_exception(exc)
            return False
        deadline = time.monotonic() + float(timeout_s)
        while time.monotonic() < deadline and not rospy.is_shutdown():
            with self.lock:
                received = bool(self.samples)
            self.write_heartbeat("subscribed")
            if self.decision is not None:
                return False
            if received:
                summary = {
                    "schema_version": 1,
                    "run_id": self.run_id,
                    "topic": topic,
                    "passed_at": _now_iso(),
                    "message_count": len(self.samples),
                    "first_sample_monotonic_ns": min(self.samples),
                    "guard_pid": self.guard_pid,
                    "guard_start_ticks": self.guard_start_ticks,
                }
                _atomic_json(self.run_dir / "resource_preflight.json", summary)
                self.stage = "awaiting_training"
                self.write_heartbeat("preflight_passed")
                return True
            time.sleep(0.1)
        self.write_decision("preflight_no_messages", {"timeout_s": float(timeout_s)})
        return False


def run_guard(args):
    guard = EgoTopicRateGuard(
        args.run_dir, args.run_id, args.window_seconds, args.minimum_rate_hz
    )
    if not guard.run_preflight(args.topic, args.preflight_timeout):
        return 2
    control_path = Path(args.run_dir) / "resource_guard_control.json"
    while True:
        try:
            control = json.loads(control_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            control = {}
        if control.get("run_id") == args.run_id and control.get("training_active"):
            guard.start_training_windows(
                control["training_started_monotonic_ns"], control["train_identity"]
            )
            break
        guard.write_heartbeat("preflight_passed")
        time.sleep(0.1)

    next_end = guard.window_start_ns + int(guard.window_seconds * 1e9)
    while True:
        try:
            control = json.loads(control_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            control = {}
        now_ns = time.monotonic_ns()
        if control.get("terminal"):
            guard.close_window(max(now_ns, guard.window_start_ns), partial=True)
            guard.stage = "terminal"
            guard.write_heartbeat("terminal")
            return 0
        if guard.decision is None and now_ns >= next_end:
            guard.close_window(next_end, partial=False)
            next_end += int(guard.window_seconds * 1e9)
            continue
        guard.write_heartbeat("stop_requested" if guard.decision else "subscribed")
        time.sleep(min(0.5, max((next_end - now_ns) / 1e9, 0.05)))


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--topic", default="/Ego_topic")
    parser.add_argument("--window-seconds", type=float, default=15.0)
    parser.add_argument("--minimum-rate-hz", type=float, default=40.0)
    parser.add_argument("--preflight-timeout", type=float, default=15.0)
    return run_guard(parser.parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
