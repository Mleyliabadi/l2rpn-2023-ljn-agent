# Copyright (c) 2023-2024 La Javaness (https://lajavaness.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of L2RPN 2023 LJN Agent, a repository for the winning agent of L2RPN 2023 competition. It is a submodule contribution to the L2RPN Baselines repository.

from grid2op.gym_compat import BoxGymObsSpace, GymEnv
from gymnasium.spaces import Discrete


class DangerBinaryPolicyTrainingEnv(GymEnv):
    def __init__(self, env_init, agent):
        super().__init__(env_init)
        # self.init_env = init_env
        self.agent = agent
        # self.g2op_last_obs = self.init_env.reset()
        # self.last_reward = None
        self.observation_space = BoxGymObsSpace(self.init_env.observation_space)
        self.action_space = Discrete(3)

    def step(self, action):
        obs = self.init_env.get_obs()
        if action == 1:
            g2op_act = self.agent.get_act_unsafe(obs, reward=0.0)
        elif action == 2:
            g2op_act = self.agent.get_act_safe(obs, reward=0.0)
        else:
            g2op_act = self.init_env.action_space()
        g2op_obs, reward, terminated, info = self.init_env.step(g2op_act)
        gym_obs = self.observation_space.to_gym(g2op_obs)
        truncated = False
        return gym_obs, float(reward), terminated, truncated, info


class UnsafeTrainingEnv(GymEnv):
    def __init__(self, env_init, agent):
        super().__init__(env_init)
        self.agent = agent

    def do_heuristic(self):
        need_action = True
        terminated = False
        info = {}
        reward = 0
        while need_action:
            obs = self.init_env.get_obs()
            act = self.agent.act(obs, reward)
            if act is not None:
                obs, reward, terminated, info = self.init_env.step(act)
                if terminated:
                    self.init_env.reset()
                    need_action = False
            else:
                need_action = False
        return obs, reward, terminated, info

    def step(self, action):
        g2op_act = self.action_space.from_gym(action)
        g2op_obs, reward, terminated, info = self.init_env.step(g2op_act)
        if not terminated:
            g2op_obs, reward, terminated, info = self.do_heuristic()
        gym_obs = self.observation_space.to_gym(g2op_obs)
        truncated = False
        return gym_obs, float(reward), terminated, truncated, info
