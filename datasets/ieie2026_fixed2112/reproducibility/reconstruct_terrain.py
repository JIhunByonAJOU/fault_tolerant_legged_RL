#!/usr/bin/env python3
"""Ubuntu + original Isaac Gym/WIM environment only; no simulation is created."""
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import isaacgym  # Must precede torch / environment imports.
import legged_gym.envs  # Establish the repository's task-registry import order.
import numpy as np

BASE = Path(__file__).resolve().parents[1]


def main():
    spec = importlib.util.spec_from_file_location('frozen_ieie_terrain', BASE / 'source_snapshot/legged_gym/utils/terrain.py')
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    for seed in [1101, 1102, 1103]:
        terrain_seed = int(hashlib.sha256(json.dumps(('terrain', seed)).encode()).hexdigest()[:8], 16) % (2**31 - 1)
        # heightfield avoids mesh conversion; the height-generation code is identical.
        cfg = SimpleNamespace(mesh_type='heightfield', terrain_length=8., terrain_width=8.,
                              terrain_proportions=[.1, .1, .35, .25, .2], num_rows=5, num_cols=12,
                              horizontal_scale=.1, vertical_scale=.005, border_size=25.,
                              curriculum=True, selected=False)
        np.random.seed(terrain_seed)
        terrain = module.Terrain(cfg, 1)
        actual = hashlib.sha256(terrain.height_field_raw.tobytes()).hexdigest()
        metadata = json.loads((BASE / f'raw/exploration_g00_s{seed}_Normal/metadata.json').read_text())
        assert actual == metadata['initial_hashes']['terrain'], (seed, actual)
        out = BASE / 'terrain'; out.mkdir(exist_ok=True)
        np.savez_compressed(out / f'heightfield_s{seed}.npz', height_field_raw=terrain.height_field_raw,
                            horizontal_scale=.1, vertical_scale=.005, world_origin_xy=np.array([-25., -25.]),
                            evaluation_seed=seed, terrain_seed=terrain_seed)
        print(seed, actual, 'MATCH')


if __name__ == '__main__': main()
