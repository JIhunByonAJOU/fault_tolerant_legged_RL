# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

import numpy as np
import os
import signal
from datetime import datetime

import isaacgym
from legged_gym.envs import *
from legged_gym.utils import class_to_dict, get_args, task_registry
import torch

def train(args):
    env, env_cfg = task_registry.make_env(name=args.task, args=args)
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args)
    wandb_run = None
    if args.wandb:
        try:
            import wandb
        except ImportError as exc:
            raise RuntimeError(
                "--wandb requires the wandb package in the active environment"
            ) from exc

        run_name = (
            os.path.basename(ppo_runner.log_dir)
            if ppo_runner.log_dir is not None
            else train_cfg.runner.run_name or None
        )
        # Keep W&B's local cache next to the checkpoints and TensorBoard files
        # for this run instead of leaving a top-level ``wandb/`` directory.
        wandb_dir = ppo_runner.log_dir or os.path.join(os.getcwd(), "logs", "wandb")
        os.makedirs(wandb_dir, exist_ok=True)
        wandb_run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=run_name,
            config={
                "task": args.task,
                "environment": class_to_dict(env_cfg),
                "training": class_to_dict(train_cfg),
            },
            # The runner uploads one metric dictionary per iteration.  The old
            # TensorBoard tailer emitted one event per scalar and fell hundreds
            # of iterations behind during the 32k-env run.
            sync_tensorboard=False,
            dir=wandb_dir,
        )
        wandb.define_metric("iteration")
        wandb.define_metric("*", step_metric="iteration")

    previous_handlers = {}

    def request_graceful_stop(signum, _frame):
        if hasattr(ppo_runner, "request_stop"):
            ppo_runner.request_stop("signal {}".format(signum))
            return
        raise KeyboardInterrupt

    for signum in (signal.SIGINT, signal.SIGTERM):
        previous_handlers[signum] = signal.getsignal(signum)
        signal.signal(signum, request_graceful_stop)

    try:
        ppo_runner.learn(
            num_learning_iterations=train_cfg.runner.max_iterations,
            init_at_random_ep_len=True,
        )
    finally:
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)
        if wandb_run is not None:
            wandb_run.finish()

if __name__ == '__main__':
    args = get_args()
    train(args)
