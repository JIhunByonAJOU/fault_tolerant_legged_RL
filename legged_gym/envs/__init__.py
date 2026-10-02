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

from legged_gym import LEGGED_GYM_ROOT_DIR, LEGGED_GYM_ENVS_DIR
from legged_gym.envs.a1.a1_config import A1RoughCfg, A1RoughCfgPPO
from .base.legged_robot import LeggedRobot
from .anymal_c.anymal import Anymal
from .anymal_c.mixed_terrains.anymal_c_rough_config import AnymalCRoughCfg, AnymalCRoughCfgPPO
from .anymal_c.flat.anymal_c_flat_config import AnymalCFlatCfg, AnymalCFlatCfgPPO
from .anymal_b.anymal_b_config import AnymalBRoughCfg, AnymalBRoughCfgPPO
from .cassie.cassie import Cassie
from .cassie.cassie_config import CassieRoughCfg, CassieRoughCfgPPO
from .a1.a1_config import A1RoughCfg, A1RoughCfgPPO
from .a1_official_wim import A1OfficialWimRoughCfg, A1OfficialWimRoughCfgPPO
from .a1_official_wim_teacher import (
    A1OfficialWimTeacher243,
    A1OfficialWimTeacher243Failure,
    A1OfficialWimTeacher243Cfg,
    A1OfficialWimTeacher243CfgPPO,
    A1OfficialWimTeacher243FailureCfg,
    A1OfficialWimTeacher243FailureCfgPPO,
    A1OfficialWimTeacher243FailureFullRangeCfg,
    A1OfficialWimTeacher243FailureFullRangeCfgPPO,
    A1OfficialWimTeacher243FailureFullRangeFromScratchCfg,
    A1OfficialWimTeacher243FailureFullRangeFromScratchCfgPPO,
    A1OfficialWimJointFailureOnset,
    A1OfficialWimJointFailureOnsetCfg,
    A1OfficialWimJointFailureOnsetCfgPPO,
    A1OfficialWimJointBetaFloorCfg,
    A1OfficialWimJointBetaFloorCfgPPO,
    A1OfficialWimFrozenTeacherStudentCfg,
    A1OfficialWimFrozenTeacherStudentCfgPPO,
    A1OfficialWimCurrentRepeatOnset,
    A1OfficialWimCurrentRepeatOnsetCfg,
    A1OfficialWimCurrentRepeatOnsetCfgPPO,
    A1OfficialWimSeparateStudentOnsetCfg,
    A1OfficialWimSeparateStudentOnsetCfgPPO,
)
from .a1_limping import (
    A1LimpingBase,
    A1LimpingBaseV2,
    A1LimpingBaseCfg,
    A1LimpingBaseCfgPPO,
    A1LimpingBaseV2Cfg,
    A1LimpingBaseV2CfgPPO,
    A1LimpingBaseWimCfg,
    A1LimpingBaseWimCfgPPO,
)


import os

from legged_gym.utils.task_registry import task_registry

task_registry.register( "anymal_c_rough", Anymal, AnymalCRoughCfg(), AnymalCRoughCfgPPO() )
task_registry.register( "anymal_c_flat", Anymal, AnymalCFlatCfg(), AnymalCFlatCfgPPO() )
task_registry.register( "anymal_b", Anymal, AnymalBRoughCfg(), AnymalBRoughCfgPPO() )
task_registry.register( "a1", LeggedRobot, A1RoughCfg(), A1RoughCfgPPO() )
task_registry.register(
    "a1_official_wim_rough",
    LeggedRobot,
    A1OfficialWimRoughCfg(),
    A1OfficialWimRoughCfgPPO(),
)
task_registry.register(
    "a1_official_wim_teacher243",
    A1OfficialWimTeacher243,
    A1OfficialWimTeacher243Cfg(),
    A1OfficialWimTeacher243CfgPPO(),
)
task_registry.register(
    "a1_official_wim_teacher243_failure",
    A1OfficialWimTeacher243Failure,
    A1OfficialWimTeacher243FailureCfg(),
    A1OfficialWimTeacher243FailureCfgPPO(),
)
task_registry.register(
    "a1_official_wim_teacher243_failure_fullrange",
    A1OfficialWimTeacher243Failure,
    A1OfficialWimTeacher243FailureFullRangeCfg(),
    A1OfficialWimTeacher243FailureFullRangeCfgPPO(),
)
task_registry.register(
    "a1_official_wim_teacher243_failure_fullrange_fromscratch",
    A1OfficialWimTeacher243Failure,
    A1OfficialWimTeacher243FailureFullRangeFromScratchCfg(),
    A1OfficialWimTeacher243FailureFullRangeFromScratchCfgPPO(),
)
task_registry.register(
    "a1_official_wim_jt_failure_fullrange_onset",
    A1OfficialWimJointFailureOnset,
    A1OfficialWimJointFailureOnsetCfg(),
    A1OfficialWimJointFailureOnsetCfgPPO(),
)
task_registry.register(
    "a1_official_wim_jt_beta_floor_onset",
    A1OfficialWimJointFailureOnset,
    A1OfficialWimJointBetaFloorCfg(),
    A1OfficialWimJointBetaFloorCfgPPO(),
)
task_registry.register(
    "a1_official_wim_frozen_tf_student_onset",
    A1OfficialWimJointFailureOnset,
    A1OfficialWimFrozenTeacherStudentCfg(),
    A1OfficialWimFrozenTeacherStudentCfgPPO(),
)
task_registry.register(
    "a1_official_wim_jt_history_free_onset",
    A1OfficialWimCurrentRepeatOnset,
    A1OfficialWimCurrentRepeatOnsetCfg(),
    A1OfficialWimCurrentRepeatOnsetCfgPPO(),
)
task_registry.register(
    "a1_official_wim_separate_student_onset",
    A1OfficialWimJointFailureOnset,
    A1OfficialWimSeparateStudentOnsetCfg(),
    A1OfficialWimSeparateStudentOnsetCfgPPO(),
)
task_registry.register(
    "a1_limping_base",
    A1LimpingBase,
    A1LimpingBaseCfg(),
    A1LimpingBaseCfgPPO(),
)
task_registry.register(
    "a1_limping_base_v2",
    A1LimpingBaseV2,
    A1LimpingBaseV2Cfg(),
    A1LimpingBaseV2CfgPPO(),
)
task_registry.register(
    "a1_limping_base_wim",
    A1LimpingBase,
    A1LimpingBaseWimCfg(),
    A1LimpingBaseWimCfgPPO(),
)
task_registry.register( "cassie", Cassie, CassieRoughCfg(), CassieRoughCfgPPO() )
