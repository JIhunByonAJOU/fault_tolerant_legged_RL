import math


def crossed_finish_plane(start_x, current_x, terrain_length=8.0):
    return float(current_x) - float(start_x) >= float(terrain_length)


def base_contact_failure(base_contact_force, threshold=1.0):
    try:
        magnitude = sum(float(value) ** 2 for value in base_contact_force) ** 0.5
    except TypeError:
        magnitude = abs(float(base_contact_force))
    return magnitude > float(threshold)


def wilson_lower_bound(successes, total, confidence=0.95, one_sided=True):
    if total <= 0 or successes < 0 or successes > total:
        raise ValueError("invalid binomial counts")
    if confidence != 0.95:
        raise ValueError("only the frozen 95 percent confidence is supported")
    z = 1.6448536269514722 if one_sided else 1.959963984540054
    p = successes / total
    denominator = 1.0 + z * z / total
    center = p + z * z / (2.0 * total)
    radius = z * math.sqrt((p * (1.0 - p) + z * z / (4.0 * total)) / total)
    return (center - radius) / denominator


def summarize_cell(episodes):
    if not episodes:
        raise ValueError("cell requires episodes")
    required = {"success", "base_contact", "progress_ratio", "command_velocity_rmse"}
    if any(set(episode) < required for episode in episodes):
        raise ValueError("incomplete episode row")
    numeric = [value for episode in episodes for value in episode.values() if isinstance(value, (int, float)) and not isinstance(value, bool)]
    if any(not math.isfinite(float(value)) for value in numeric):
        raise ValueError("non-finite episode row")
    successes = sum(bool(item["success"]) for item in episodes)
    ordered_progress = sorted(float(item["progress_ratio"]) for item in episodes)
    return {
        "episodes": len(episodes),
        "successes": successes,
        "success_rate": successes / len(episodes),
        "base_contact_rate": sum(bool(item["base_contact"]) for item in episodes) / len(episodes),
        "median_progress_ratio": ordered_progress[len(ordered_progress) // 2],
        "command_velocity_rmse": math.sqrt(sum(float(item["command_velocity_rmse"]) ** 2 for item in episodes) / len(episodes)),
        "finite": True,
    }
