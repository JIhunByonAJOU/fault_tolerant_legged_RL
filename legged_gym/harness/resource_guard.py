"""Local ROS topic-rate evidence collector for managed training runs."""

import argparse
from collections import deque
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

    def __init__(
        self,
        run_dir,
        run_id,
        window_seconds=15.0,
        minimum_rate_hz=40.0,
        long_gap_threshold_seconds=1.0,
    ):
        self.run_dir = Path(run_dir)
        self.run_id = str(run_id)
        self.window_seconds = float(window_seconds)
        self.minimum_rate_hz = float(minimum_rate_hz)
        self.long_gap_threshold_seconds = float(long_gap_threshold_seconds)
        if self.long_gap_threshold_seconds <= 0.0:
            raise ValueError("long gap threshold must be positive")
        self.guard_pid = os.getpid()
        self.guard_start_ticks = _proc_start_ticks(self.guard_pid)
        self.samples = []
        self.lock = threading.Lock()
        self.stage = "preflight"
        self.window_start_ns = None
        self.consecutive_below = 0
        self.recovery_required = False
        self.last_sample_ns = None
        self.gap_events = deque()
        self.seen_gap_event_ids = set()
        self.gap_event_ledger_limit = 256
        self.open_gap_provisional_id = None
        self.open_gap_start_ns = None
        self.pending_gap_event_id = None
        self.pending_gap_end_ns = None
        self.window_sequence = 0
        self.subscriber_status = "starting"
        self.train_identity = {}
        self.decision = None

    def _register_gap_event_locked(self, left_ns, right_ns):
        if right_ns <= left_ns:
            return None
        duration_s = (right_ns - left_ns) / 1e9
        if duration_s < self.long_gap_threshold_seconds:
            return None
        event_id = "{}:{}".format(left_ns, right_ns)
        if event_id in self.seen_gap_event_ids:
            return None
        provisional_id = None
        continues_pending = False
        if self.open_gap_start_ns == left_ns:
            provisional_id = self.open_gap_provisional_id
            continues_pending = self.pending_gap_event_id == provisional_id
        event = {
            "schema_version": 3,
            "run_id": self.run_id,
            "gap_event_id": event_id,
            "start_monotonic_ns": left_ns,
            "end_monotonic_ns": right_ns,
            "duration_s": duration_s,
            "recorded_at": _now_iso(),
            "assigned": False,
            "provisional_gap_event_id": provisional_id,
            "continues_pending": continues_pending,
        }
        if len(self.gap_events) >= self.gap_event_ledger_limit:
            expired = self.gap_events.popleft()
            self.seen_gap_event_ids.discard(expired["gap_event_id"])
        self.gap_events.append(event)
        self.seen_gap_event_ids.add(event_id)
        if continues_pending:
            self.pending_gap_event_id = event_id
            self.pending_gap_end_ns = right_ns
        if provisional_id is not None:
            self.open_gap_provisional_id = None
            self.open_gap_start_ns = None
        return event

    def _append_gap_event(self, event):
        durable_event = {
            key: value
            for key, value in event.items()
            if key not in {"assigned", "continues_pending"}
        }
        _append_jsonl(self.run_dir / "resource_gap_events.jsonl", durable_event)

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
        gap_event = None
        with self.lock:
            self.samples.append(monotonic_ns)
            if self.stage == "running":
                previous_ns = self.last_sample_ns
                if previous_ns is None or monotonic_ns > previous_ns:
                    self.last_sample_ns = monotonic_ns
                if previous_ns is not None and monotonic_ns > previous_ns:
                    gap_event = self._register_gap_event_locked(previous_ns, monotonic_ns)
        _append_jsonl(self.run_dir / "resource_samples.jsonl", record)
        if gap_event is not None:
            self._append_gap_event(gap_event)
        return record

    def start_training_windows(self, start_ns, train_identity):
        start_ns = int(start_ns)
        retained_gap_events = []
        with self.lock:
            self.stage = "running"
            self.window_start_ns = start_ns
            self.train_identity = dict(train_identity)
            self.samples = [value for value in self.samples if value >= start_ns]
            self.last_sample_ns = max(self.samples, default=None)
            self.gap_events.clear()
            self.seen_gap_event_ids.clear()
            self.open_gap_provisional_id = None
            self.open_gap_start_ns = None
            self.pending_gap_event_id = None
            self.pending_gap_end_ns = None
            self.recovery_required = False
            self.consecutive_below = 0
            retained = sorted(set(self.samples))
            for left_ns, right_ns in zip(retained, retained[1:]):
                event = self._register_gap_event_locked(left_ns, right_ns)
                if event is not None:
                    retained_gap_events.append(event)
        for event in retained_gap_events:
            self._append_gap_event(event)
        self.write_heartbeat("subscribed")

    def close_window(self, end_ns, partial=False):
        if self.window_start_ns is None:
            raise ValueError("training window has not started")
        end_ns = int(end_ns)
        if end_ns < self.window_start_ns:
            raise ValueError("window end precedes start")
        start_ns = self.window_start_ns
        duration_s = (end_ns - start_ns) / 1e9
        decision_reason = None
        event = None
        gap_event_disposition = None
        recovery_eligible = False
        open_gap_age_s = None
        provisional_gap_event_id = None
        with self.lock:
            latest_arrival_at_or_before_end = max(
                (value for value in self.samples if value <= end_ns),
                default=None,
            )
            arrivals = sorted(
                value for value in self.samples if start_ns <= value < end_ns
            )
            self.samples = [value for value in self.samples if value >= end_ns]
            boundaries = [start_ns] + arrivals + [end_ns]
            max_gap_s = max(
                ((right - left) / 1e9 for left, right in zip(boundaries, boundaries[1:])),
                default=duration_s,
            )
            internal_max_gap_s = max(
                (
                    (right - left) / 1e9
                    for left, right in zip(arrivals, arrivals[1:])
                ),
                default=0.0,
            )
            rate_hz = len(arrivals) / duration_s if duration_s > 0 else 0.0
            complete = not partial
            if (
                complete
                and arrivals
                and self.subscriber_status == "subscribed"
                and latest_arrival_at_or_before_end is not None
            ):
                open_gap_age_s = (end_ns - latest_arrival_at_or_before_end) / 1e9
            open_gap_start_ns = None
            if (
                open_gap_age_s is not None
                and open_gap_age_s >= self.long_gap_threshold_seconds
            ):
                open_gap_start_ns = latest_arrival_at_or_before_end
            new_events = []
            if complete:
                for candidate in self.gap_events:
                    if not candidate["assigned"] and candidate["end_monotonic_ns"] <= end_ns:
                        candidate["assigned"] = True
                        new_events.append(candidate)

            pending_start_ns = self.open_gap_start_ns
            if pending_start_ns is None and self.pending_gap_event_id:
                pending_identity = str(self.pending_gap_event_id)
                if pending_identity.startswith("open:"):
                    pending_identity = pending_identity[len("open:") :]
                else:
                    pending_identity = pending_identity.split(":", 1)[0]
                try:
                    pending_start_ns = int(pending_identity)
                except ValueError:
                    pending_start_ns = None
            for candidate in new_events:
                same_pending_start = (
                    self.recovery_required
                    and pending_start_ns is not None
                    and candidate["start_monotonic_ns"] == pending_start_ns
                )
                candidate["same_pending_at_close"] = same_pending_start
                if same_pending_start:
                    provisional_id = "open:{}".format(pending_start_ns)
                    candidate["provisional_gap_event_id"] = provisional_id
                    self.pending_gap_event_id = candidate["gap_event_id"]
                    self.pending_gap_end_ns = candidate["end_monotonic_ns"]
                    self.open_gap_provisional_id = None
                    self.open_gap_start_ns = None

            if not complete:
                rate_gate_status = "partial"
            elif not arrivals:
                rate_gate_status = "no_message_window"
                decision_reason = "no_message_window"
            elif new_events:
                continuation_events = [
                    candidate
                    for candidate in new_events
                    if candidate.get("same_pending_at_close")
                ]
                distinct_events = [
                    candidate
                    for candidate in new_events
                    if not candidate.get("same_pending_at_close")
                ]
                if self.recovery_required and continuation_events and not distinct_events:
                    event = continuation_events[0]
                    provisional_gap_event_id = event.get("provisional_gap_event_id")
                    rate_gate_status = "external_gap_continuation"
                    gap_event_disposition = "continuation"
                    self.consecutive_below = 0
                elif self.recovery_required:
                    event = distinct_events[0] if distinct_events else new_events[-1]
                    provisional_gap_event_id = event.get("provisional_gap_event_id")
                    rate_gate_status = "external_gap_not_recovered"
                    gap_event_disposition = "distinct_gap_before_recovery"
                    decision_reason = "external_gap_not_recovered"
                else:
                    event = new_events[0]
                    provisional_gap_event_id = event.get("provisional_gap_event_id")
                    rate_gate_status = "external_gap_candidate"
                    gap_event_disposition = "candidate"
                    self.pending_gap_event_id = event["gap_event_id"]
                    self.pending_gap_end_ns = event["end_monotonic_ns"]
                    self.recovery_required = True
                    self.consecutive_below = 0
                    if len(new_events) > 1:
                        event = new_events[1]
                        rate_gate_status = "external_gap_not_recovered"
                        gap_event_disposition = "distinct_gap_before_recovery"
                        decision_reason = "external_gap_not_recovered"
                if open_gap_start_ns is not None:
                    provisional_gap_event_id = "open:{}".format(open_gap_start_ns)
                    event = {
                        "gap_event_id": None,
                        "start_monotonic_ns": open_gap_start_ns,
                        "end_monotonic_ns": None,
                        "duration_s": open_gap_age_s,
                    }
                    rate_gate_status = "external_gap_not_recovered"
                    gap_event_disposition = "distinct_open_gap_before_recovery"
                    decision_reason = "external_gap_not_recovered"
                    self.open_gap_provisional_id = provisional_gap_event_id
                    self.open_gap_start_ns = open_gap_start_ns
            elif open_gap_start_ns is not None:
                provisional_gap_event_id = "open:{}".format(open_gap_start_ns)
                event = {
                    "gap_event_id": None,
                    "start_monotonic_ns": open_gap_start_ns,
                    "end_monotonic_ns": None,
                    "duration_s": open_gap_age_s,
                }
                if self.recovery_required:
                    rate_gate_status = "external_gap_not_recovered"
                    gap_event_disposition = "distinct_open_gap_before_recovery"
                    decision_reason = "external_gap_not_recovered"
                else:
                    rate_gate_status = "external_gap_open_candidate"
                    gap_event_disposition = "open_candidate"
                    self.pending_gap_event_id = provisional_gap_event_id
                    self.pending_gap_end_ns = None
                    self.recovery_required = True
                    self.consecutive_below = 0
                self.open_gap_provisional_id = provisional_gap_event_id
                self.open_gap_start_ns = open_gap_start_ns
            elif self.recovery_required:
                event = next(
                    (
                        candidate
                        for candidate in reversed(self.gap_events)
                        if candidate["gap_event_id"] == self.pending_gap_event_id
                    ),
                    None,
                )
                recovery_eligible = (
                    self.pending_gap_end_ns is not None
                    and start_ns >= self.pending_gap_end_ns
                    and internal_max_gap_s < self.long_gap_threshold_seconds
                )
                if not recovery_eligible:
                    rate_gate_status = "external_gap_continuation"
                    gap_event_disposition = "continuation"
                elif rate_hz >= self.minimum_rate_hz:
                    rate_gate_status = "external_gap_recovered"
                    gap_event_disposition = "recovered"
                    self.consecutive_below = 0
                    self.recovery_required = False
                    self.pending_gap_event_id = None
                    self.pending_gap_end_ns = None
                else:
                    rate_gate_status = "external_gap_not_recovered"
                    gap_event_disposition = "recovery_below_rate"
                    decision_reason = "external_gap_not_recovered"
            elif rate_hz < self.minimum_rate_hz:
                rate_gate_status = "gap_free_below_rate"
                self.consecutive_below += 1
                if self.consecutive_below >= 2:
                    decision_reason = "two_consecutive_gap_free_below_rate"
            else:
                rate_gate_status = "gap_free_rate_ok"
                self.consecutive_below = 0

            record = {
                "schema_version": 3,
                "run_id": self.run_id,
                "sequence": self.window_sequence,
                "window_start_monotonic_ns": start_ns,
                "window_end_monotonic_ns": end_ns,
                "duration_s": duration_s,
                "message_count": len(arrivals),
                "rate_hz": rate_hz,
                "max_gap_s": max_gap_s,
                "complete": complete,
                "partial": bool(partial),
                "consecutive_below_40": self.consecutive_below,
                "rate_gate_status": rate_gate_status,
                "long_gap_threshold_s": self.long_gap_threshold_seconds,
                "gap_free_consecutive_below_40": self.consecutive_below,
                "recovery_required": self.recovery_required,
                "recovery_eligible": recovery_eligible,
                "provisional_gap_event_id": provisional_gap_event_id,
                "gap_event_id": event.get("gap_event_id") if event else None,
                "gap_event_start_monotonic_ns": (
                    event["start_monotonic_ns"] if event else None
                ),
                "gap_event_end_monotonic_ns": (
                    event["end_monotonic_ns"] if event else None
                ),
                "gap_event_duration_s": event["duration_s"] if event else None,
                "gap_event_disposition": gap_event_disposition,
                "open_gap_age_s": open_gap_age_s,
                "subscriber_status": self.subscriber_status,
                "guard_pid": self.guard_pid,
                "guard_start_ticks": self.guard_start_ticks,
                "train_pid": self.train_identity.get("train_pid"),
                "train_start_ticks": self.train_identity.get("train_start_ticks"),
            }
            self.window_sequence += 1
            self.window_start_ns = end_ns
        end_wall = time.time()
        start_wall = end_wall - duration_s
        record["window_start"] = dt.datetime.fromtimestamp(start_wall).astimezone().isoformat(
            timespec="microseconds"
        )
        record["window_end"] = dt.datetime.fromtimestamp(end_wall).astimezone().isoformat(
            timespec="microseconds"
        )
        _append_jsonl(self.run_dir / "resource_windows.jsonl", record)
        if decision_reason is not None:
            self.write_decision(decision_reason, record)
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
        args.run_dir,
        args.run_id,
        args.window_seconds,
        args.minimum_rate_hz,
        args.long_gap_threshold_seconds,
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
    parser.add_argument("--long-gap-threshold-seconds", type=float, default=1.0)
    parser.add_argument("--preflight-timeout", type=float, default=15.0)
    return run_guard(parser.parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
