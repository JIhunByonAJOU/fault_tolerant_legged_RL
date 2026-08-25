import argparse
import json
import os
import tempfile
from pathlib import Path

from legged_gym.official_wim import conformance


def _atomic(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix="." + path.name + ".", dir=str(path.parent))
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, allow_nan=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--official-legged-gym-commit", required=True)
    parser.add_argument("--rsl-rl-root", required=True)
    parser.add_argument("--official-rsl-rl-commit", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    if args.official_legged_gym_commit != conformance.OFFICIAL_LEGGED_GYM_COMMIT:
        raise SystemExit("unexpected official legged_gym revision")
    if Path(args.rsl_rl_root).resolve() != conformance.RSL_RL_ROOT.resolve() or args.official_rsl_rl_commit != conformance.OFFICIAL_RSL_RL_COMMIT:
        raise SystemExit("unexpected official rsl_rl source")
    result = conformance.complete_conformance(args.device)
    _atomic(args.output, result)
    print(json.dumps(result, sort_keys=True))
    return 0 if result["pass"] or not args.strict else 2


if __name__ == "__main__":
    raise SystemExit(main())
