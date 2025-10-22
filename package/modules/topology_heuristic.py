# Copyright (c) 2023-2024 La Javaness (https://lajavaness.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of L2RPN 2023 LJN Agent, a repository for the winning agent of L2RPN 2023 competition. It is a submodule contribution to the L2RPN Baselines repository.

import numpy as np
from grid2op.Action import ActionSpace
from grid2op.Agent import RecoPowerlineAgent, RecoPowerlinePerArea

from .base_module import BaseModule, GreedyModule
from .utils import load_action_to_grid2op


class RecoPowerlineModule(RecoPowerlineAgent, BaseModule):
    """Module wrapper for the greedy RecoPowerlineAgent from Grid2Op.
    This module will try to best reconnection possible at each time step.
    """

    def __init__(self, action_space: ActionSpace):
        RecoPowerlineAgent.__init__(self, action_space)
        BaseModule.__init__(self, action_space)

    def get_act(self, observation, base_action, reward, done=False):
        return self.act(observation, reward)


class RecoPowerlinePerAreaModule(RecoPowerlinePerArea, BaseModule):
    """Module wrapper for the greedy RecoPowerlinePerArea from Grid2Op.
    This module will try to reconnect powerlines in each sub-area of the environment.
    """

    def __init__(
        self, action_space: ActionSpace, areas_by_sub_id: dict, lines_id_by_area: dict
    ):
        RecoPowerlinePerArea.__init__(self, action_space, areas_by_sub_id)
        BaseModule.__init__(self, action_space)
        self.lines_in_area = [list_ids for list_ids in lines_id_by_area.values()]

    def get_act(self, observation, base_action, reward, done=False):
        # Initialize line status info
        line_stat_s = observation.line_status
        cooldown = observation.time_before_cooldown_line
        maintenance = observation.time_next_maintenance
        can_be_reco = (~line_stat_s) & (cooldown == 0) & (maintenance != 1)

        # Initialize variables to store best actions and their results for each area.
        min_rho_per_area = [np.inf for _ in self.lines_in_area]
        action_chosen_per_area = [
            self.action_space({}) for _ in self.lines_in_area
        ]  # Default to "do nothing" actions
        obs_simu_chosen_per_area = [None for _ in self.lines_in_area]

        if not can_be_reco.any():
            # no line to reconnect
            return self.action_space()
        # If there are lines that can be reconnected, simulate their reconnection.
        actions_and_ids = [
            (self.action_space({"set_line_status": [(id_, +1)]}), id_)
            for id_ in np.where(can_be_reco)[0]
        ]
        for action, id_ in actions_and_ids:
            obs_simu, _reward, _done, _info = observation.simulate(
                action + base_action, time_step=1
            )

            # Determine the area of the line using lines_in_area.
            area_id = next(
                (idx for idx, lines in enumerate(self.lines_in_area) if id_ in lines),
                None,
            )

            # If the simulation is successful and the result is better than previous simulations for the area,
            # store this action as the best action for that area.
            if area_id is not None and (
                obs_simu.rho.max() < min_rho_per_area[area_id]
            ) & (len(_info["exception"]) == 0):
                action_chosen_per_area[area_id] = action
                obs_simu_chosen_per_area[area_id] = obs_simu
                min_rho_per_area[area_id] = obs_simu.rho.max()

        reco_act = self.action_space()
        for act in action_chosen_per_area:
            reco_act += act
        return reco_act


class RecoverInitTopoModule(GreedyModule):
    """Module for initial topology recovering.
    This module will perform the best action to recover initial topology by changing bus
    (single action, do not support multiple sub-zone actions)
    """

    def __init__(self, action_space: ActionSpace):
        GreedyModule.__init__(self, action_space)

    def _get_tested_action(self, observation):
        # Get the list of possible actions to revert the grid's topology to its reference state.
        tested_action = self.action_space.get_back_to_ref_state(observation).get(
            "substation", None
        )
        if tested_action is not None:
            tested_action = [
                act
                for act in tested_action
                if (
                    observation.time_before_cooldown_sub[
                        int(act.as_dict()["set_bus_vect"]["modif_subs_id"][0])
                    ]
                    == 0
                )
            ]
            return tested_action
        return []


class TopoSearchModule(GreedyModule):
    def __init__(self, action_space: ActionSpace, action_vec_path: str):
        GreedyModule.__init__(self, action_space)
        BaseModule.__init__(self, action_space)
        self.topo_act_list = []
        self.load_action_space(action_vec_path)

    def load_action_space(self, action_vec_path: str):
        self.topo_act_list += load_action_to_grid2op(self.action_space, action_vec_path)

    def _get_tested_action(self, observation):
        return [
            act
            for act in self.topo_act_list
            if (
                observation.time_before_cooldown_sub[
                    int(act.as_dict()["set_bus_vect"]["modif_subs_id"][0])
                ]
                == 0
            )
        ]


class ChallengeTopoSearchModule(BaseModule):
    def __init__(
        self,
        action_space: ActionSpace,
        topo_n1_unsafe: TopoSearchModule,
        topo_12_unsafe: TopoSearchModule,
        recover_topo: RecoverInitTopoModule,
        rho_danger: float = 0.99,
    ):
        BaseModule.__init__(self, action_space=action_space)
        self.topo_12_unsafe = topo_12_unsafe
        self.topo_n1_unsafe = topo_n1_unsafe
        self.recover_topo = recover_topo
        self.rho_danger = rho_danger

    def get_act(self, observation, base_action, reward, done=False):
        act = base_action
        # Try to perform a topology recovery at first (= reconnection to bus 1)
        recovery_act = self.recover_topo.get_act(
            observation, base_action, reward, rho_threshold=self.rho_danger
        )
        if recovery_act is not None:
            return recovery_act

        else:
            # Case 1 : All lines are connected
            if all(observation.line_status):
                topo_act = self.topo_12_unsafe.get_act(observation, base_action, reward)
            # Case 2 : N-1 situation, at least one line is disconnected
            else:
                topo_act = self.topo_n1_unsafe.get_act(observation, base_action, reward)

            if topo_act is not None:
                return topo_act

    def act(self, observation, reward, done=False):
        act = self.get_act(observation, self.action_space(), reward)
        return act if act is not None else self.action_space()


class ZoneBasedTopoSearchModule(TopoSearchModule):
    def __init__(
        self,
        action_space: ActionSpace,
        action_vec_path: str,
        areas_by_sub_id: dict,
        line_to_sub_id: list,
    ):
        super().__init__(action_space, action_vec_path)
        self.areas_by_sub_id = {}
        for key, value in areas_by_sub_id.items():
            for id in value:
                self.areas_by_sub_id[id] = key
        self.line_to_sub_id = line_to_sub_id

        self._init_zone_based_action_dict()

    def _init_zone_based_action_dict(self):
        self.topo_act_list_by_area = {i: [] for i in range(3)}
        for act in self.topo_act_list:
            sub_id = int(act.as_dict()["set_bus_vect"]["modif_subs_id"][0])
            area = self.areas_by_sub_id[sub_id]
            self.topo_act_list_by_area[area].append(act)

    def _get_tested_action(self, observation, area: int = None):
        if area is None:
            area = self.areas_by_sub_id[self.line_to_sub_id[np.argmax(observation.rho)]]
        return [
            act
            for act in self.topo_act_list_by_area[area]
            if (
                observation.time_before_cooldown_sub[
                    int(act.as_dict()["set_bus_vect"]["modif_subs_id"][0])
                ]
                == 0
            )
        ]
