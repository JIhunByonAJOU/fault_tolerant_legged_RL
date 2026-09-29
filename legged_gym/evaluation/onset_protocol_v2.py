"""Simulator-independent protocol definitions for diagnostic onset evaluation."""
import random


def make_specs(seed, repeats, rates, dt=0.02, onset_range=(2.0, 10.0)):
    if repeats < 1 or not rates or any(not 0 <= d <= 1 for d in rates):
        raise ValueError('Invalid conditions')
    low, high = [round(t / dt) for t in onset_range]
    if low < 0 or high < low:
        raise ValueError('Invalid onset range')
    rng = random.Random(seed)
    # Match onset draws across severities for each joint/replicate.
    return [(j, d, rng_step, rep)
            for rep in range(repeats) for j in range(12)
            for rng_step in [rng.randint(low, high)] for d in rates]


def tracking_summary(stable, alive, window_steps):
    """Reference used by tests; dead time is part of the fixed denominator."""
    if len(stable) != len(alive) or not stable or window_steps < 1:
        raise ValueError('Invalid sample sequence')
    run, recovered, count = 0, False, 0
    for ok, living in zip(stable, alive):
        ok = bool(ok and living)
        count += ok
        run = run + 1 if ok else 0
        recovered |= run >= window_steps
    return count / len(stable), recovered
