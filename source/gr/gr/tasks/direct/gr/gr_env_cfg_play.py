# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.envs import ViewerCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass

from .gr_env_cfg import GrEnvCfg as TrainGrEnvCfg


@configclass
class GrEnvCfg(TrainGrEnvCfg):
    """Play-time config that reuses the simplified training environment."""

    play = True

    viewer: ViewerCfg = ViewerCfg(
        eye=(1.6, -1.2, 1.3),
        lookat=(0.0, 0.0, 0.45),
    )

    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=4,
        env_spacing=2.5,
        replicate_physics=True,
        clone_in_fabric=False,
    )
