# Copyright (c) 2023-2024 La Javaness (https://lajavaness.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of L2RPN 2023 LJN Agent, a repository for the winning agent of L2RPN 2023 competition. It is a submodule contribution to the L2RPN Baselines repository.

from gymnasium.spaces import Discrete


class GlobalTopoActionSpace(Discrete):
    """
    Simple action space encoding a reduced topo action space a Discrete gym space.
    An index corresponds to a single topological action.
    """

    def __init__(
        self,
        topo_actions_list: list,
        g2op_action_space,
        is_encoded_curriculum: bool = False,
    ):
        Discrete.__init__(self, len(topo_actions_list))
        self.topo_actions_list = topo_actions_list
        self.g2op_action_space = g2op_action_space

    def from_gym(self, gym_action):
        return self.topo_actions_list[int(gym_action)]

    def close(self):
        if hasattr(self, "_init_env"):
            self._init_env = None  # this doesn't own the environment
