# Copyright (c) 2023-2024 La Javaness (https://lajavaness.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of L2RPN 2023 LJN Agent, a repository for the winning agent of L2RPN 2023 competition. It is a submodule contribution to the L2RPN Baselines repository.

from grid2op.Action import BaseAction
from grid2op.dtypes import dt_float
from grid2op.Environment import Environment
from grid2op.Reward import BaseReward


class MaxRhoReward(BaseReward):
    def __init__(self, logger=None):
        BaseReward.__init__(self, logger=logger)
        self.reward_min = dt_float(-1.0)
        self.reward_max = dt_float(1.0)

    def __call__(self, action, env, has_error, is_done, is_illegal, is_ambiguous):
        if has_error or is_illegal or is_ambiguous:
            return self.reward_min

        obs = env.get_obs(_do_copy=False)
        return 2.0 - obs.rho.max()


class PPO_Reward(BaseReward):
    def __init__(self):
        """
        PPO_Reward class, based on the BaseReward from Grid2Op
        """
        BaseReward.__init__(self)
        self.reward_min = -10
        self.reward_std = 2

    def __call__(
        self,
        action: BaseAction,
        env: Environment,
        has_error: bool,
        is_done: bool,
        is_illegal: bool,
        is_ambiguous: bool,
    ) -> float:
        if is_done or is_illegal or is_ambiguous or has_error:
            return self.reward_min
        rho_max = env.get_obs().rho.max()

        action_contrib = 0.0
        if action == env.action_space():
            action_contrib = 4.0
        return self.reward_std - rho_max * (1 if rho_max < 0.95 else 2) + action_contrib
