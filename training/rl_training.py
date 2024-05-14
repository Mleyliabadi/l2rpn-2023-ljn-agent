# Copyright (c) 2023-2024 La Javaness (https://lajavaness.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of L2RPN 2023 LJN Agent, a repository for the winning agent of L2RPN 2023 competition. It is a submodule contribution to the L2RPN Baselines repository.s

import os

import grid2op
import numpy as np
from grid2op.gym_compat import BoxGymObsSpace, GymEnv
from lightsim2grid.lightSimBackend import LightSimBackend
from stable_baselines3 import PPO

from ..agent import AgentTopoNN
from ..gym_assets.action_space import GlobalTopoActionSpace
from ..gym_assets.environment import UnsafeTrainingEnv
from ..modules.rewards import PPO_Reward
from ..utils import NN_ACT_SPACE_DIR

if __name__ == "__main__":
    # Grid2Op env
    env = grid2op.make(
        "l2rpn_idf_2023", backend=LightSimBackend(), reward_class=PPO_Reward
    )

    # Instantiating gym environment. Observation and act space should match spaces used during supervised training.
    attr_to_keep = ["rho"]  # Provided as a baseline working example
    obs_space = BoxGymObsSpace(env.observation_space, attr_to_keep)
    act_space_data = np.load(
        os.path.join(NN_ACT_SPACE_DIR, "action_12_unsafe_nn.npz"), allow_pickle=True
    )["g2op_id_actions"]
    act_space = GlobalTopoActionSpace(act_space_data, env.action_space)

    gym_env = GymEnv(env)
    gym_env.observation_space.close()
    gym_env.observation_space = obs_space
    gym_env.action_space.close()
    gym_env.action_space = act_space

    model_path = "/home/jules/RTE/ai-dispatcher-agent/models/12_unsafe_rho_overflow.zip"

    agent = AgentTopoNN(
        env.action_space, env, gym_env, model_path=model_path, training_mode=True
    )
    model = PPO.load(model_path)

    # Instantiating specific traininf env.
    # This env emulates the behavior of the complete agent except for the part that should be trained by reinforcement.
    training_env = UnsafeTrainingEnv(env, agent)
    training_env.observation_space.close()
    training_env.observation_space = obs_space
    training_env.action_space.close()
    training_env.action_space = act_space

    model.set_env(training_env)

    # Train
    model.learn(10000)

    # Save
    model.save("RL_training_PPO.zip")
