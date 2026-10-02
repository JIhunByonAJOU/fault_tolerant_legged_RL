"""Shared CUDA-drain pacing used by the JT, B1, and B2 training paths."""

import json
import math
import os
import time
from pathlib import Path

import torch


def validate_iteration_sleep_ms(value):
    """Return a finite, nonnegative iteration-boundary sleep in milliseconds."""
    value = float(value)
    if not math.isfinite(value) or value < 0.0:
        raise ValueError(
            "shared_gpu_iteration_sleep_ms must be finite and nonnegative"
        )
    return value


def synchronize_active_cuda(device):
    """Drain the active CUDA device and return the monotonic elapsed seconds."""
    device = torch.device(device)
    if device.type != "cuda":
        return 0.0
    started = time.monotonic()
    torch.cuda.synchronize(device)
    return time.monotonic() - started


class SharedGpuIterationPacer:
    """Apply and durably record post-iteration CUDA-idle intervals."""

    def __init__(self, sleep_ms, device, log_dir=None):
        self.sleep_ms = validate_iteration_sleep_ms(sleep_ms)
        self.device = device
        self.log_path = (
            Path(log_dir) / "iteration_boundary_pacing.jsonl"
            if log_dir is not None
            else None
        )

    def pause_after_completed_iteration(
        self,
        completed_iteration,
        next_iteration,
        active_iteration_seconds,
        stop_requested=None,
    ):
        """Synchronize, sleep responsively, and append one durable pacing row."""
        if self.sleep_ms <= 0.0:
            return None
        stop_requested = stop_requested or (lambda: False)
        if stop_requested():
            return None

        sync_seconds = synchronize_active_cuda(self.device)
        pause_start_ns = time.monotonic_ns()
        deadline_ns = pause_start_ns + int(self.sleep_ms * 1.0e6)
        while not stop_requested():
            remaining_seconds = (deadline_ns - time.monotonic_ns()) / 1.0e9
            if remaining_seconds <= 0.0:
                break
            time.sleep(min(remaining_seconds, 0.05))
        pause_end_ns = time.monotonic_ns()
        observed_sleep_seconds = (pause_end_ns - pause_start_ns) / 1.0e9
        record = {
            "completed_iteration": int(completed_iteration),
            "cuda_sync_seconds": float(sync_seconds),
            "requested_sleep_ms": float(self.sleep_ms),
            "observed_sleep_seconds": float(observed_sleep_seconds),
            "pause_start_monotonic_ns": int(pause_start_ns),
            "pause_end_monotonic_ns": int(pause_end_ns),
            "next_iteration": int(next_iteration),
            "Perf/iteration_cycle_time": float(
                active_iteration_seconds + sync_seconds + observed_sleep_seconds
            ),
        }
        if self.log_path is not None:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            with self.log_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, sort_keys=True) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
        return record
