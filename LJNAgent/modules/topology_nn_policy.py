# Copyright (c) 2023-2024 La Javaness (https://lajavaness.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of L2RPN 2023 LJN Agent, a repository for the winning agent of L2RPN 2023 competition. It is a submodule contribution to the L2RPN Baselines repository.

import torch
from stable_baselines3.ppo import PPO

from .base_module import GreedyModule


class TopoNNTopKModule(GreedyModule):
    def __init__(
        self,
        action_space,
        gym_env,
        model_path: str,
        top_k: int = 1,
        device: str = "cpu",
    ):
        GreedyModule.__init__(self, action_space)
        self.top_k = top_k
        self.device = device
        self.gym_env = gym_env
        self.model = None
        self.load_policy(model_path)

    def get_top_k(self, gym_obs, top_k: int):
        input = torch.from_numpy(gym_obs).reshape((1, len(gym_obs))).to(self.device)
        distribution = self.model.policy.get_distribution(input)
        logits = distribution.distribution.logits
        return torch.topk(logits, k=top_k)[1].cpu().numpy()[0]

    def load_policy(self, model_path: str):
        self.model = PPO.load(model_path, device=self.device, custom_objects = {'observation_space' : self.gym_env.observation_space, 'action_space' : self.gym_env.action_space})

    def _get_tested_action(self, observation):
        gym_obs = self.gym_env.observation_space.to_gym(observation)
        act_id_list = self.get_top_k(gym_obs, top_k=self.top_k)
        return [self.gym_env.action_space.from_gym(i) for i in act_id_list]
