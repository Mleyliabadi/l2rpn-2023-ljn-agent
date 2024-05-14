# Copyright (c) 2023-2024 La Javaness (https://lajavaness.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of L2RPN 2023 LJN Agent, a repository for the winning agent of L2RPN 2023 competition. It is a submodule contribution to the L2RPN Baselines repository.

# from stable_baselines3.common import set_global_seeds
import logging

import grid2op
from dispatcher_agent.agent import DangerPolicyAgent
from dispatcher_agent.gym_assets.environment import \
    DangerBinaryPolicyTrainingEnv
from dispatcher_agent.modules.rewards import MaxRhoReward, PPO_Reward
from grid2op.gym_compat import BoxGymObsSpace, GymEnv
from gymnasium.spaces import Discrete
from lightsim2grid.lightSimBackend import LightSimBackend
from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv


def instantiate_env():
    env = grid2op.make(
        "l2rpn_idf_2023", backend=LightSimBackend(), reward_class=PPO_Reward
    )
    env.seed(0)
    agent = DangerPolicyAgent(env.action_space, env)
    env_gym = DangerBinaryPolicyTrainingEnv(env, agent)
    return env_gym


def make_env(env_id, rank, seed=0):
    """
    Utility function for multiprocessed env.

    :param env_id: (str) the environment ID
    :param num_env: (int) the number of environments you wish to have in subprocesses
    :param seed: (int) the inital seed for RNG
    :param rank: (int) index of the subprocess
    """

    def _init():
        env = grid2op.make(
            "l2rpn_idf_2023", backend=LightSimBackend(), reward_class=PPO_Reward
        )
        env.seed(seed + rank)
        agent = DangerPolicyAgent(env.action_space, env)
        env_gym = DangerBinaryPolicyTrainingEnv(env, agent)
        return env_gym

    # set_global_seeds(seed)
    return _init


def main():
    from dispatcher_agent.utils import DATA_DIR

    grid2op.change_local_dir(DATA_DIR)
    env_id = "l2rpn_idf_2023"
    num_cpu = 128  # Number of processes to use
    # Create the vectorized environment
    env = SubprocVecEnv([make_env(env_id, i) for i in range(num_cpu)])

    model = PPO("MlpPolicy", env, n_steps=2048)
    model.learn(total_timesteps=1e5, progress_bar=True)
    model.save("model_rl_highlevelpolicy.zip")

    obs = env.reset()
    for _ in range(1000):
        action, _states = model.predict(obs)
        obs, rewards, dones, info = env.step(action)
        env.render()


if __name__ == "__main__":

    main()
