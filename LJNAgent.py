# Copyright (c) 2023-2024 La Javaness (https://lajavaness.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of L2RPN 2023 LJN Agent, a repository for the winning agent of L2RPN 2023 competition. It is a submodule contribution to the L2RPN Baselines repository.

import copy
import logging
import os
import time
from abc import ABC, abstractmethod

from grid2op.Action import ActionSpace, BaseAction
from grid2op.Agent import BaseAgent
from grid2op.gym_compat import BoxGymObsSpace
from grid2op.Observation import BaseObservation
from stable_baselines3 import PPO

from .modules.base_module import BaseModule
from .modules.convex_optim import OptimModule
from .modules.topology_heuristic import (ChallengeTopoSearchModule,
                                         RecoPowerlinePerAreaModule,
                                         RecoverInitTopoModule,
                                         TopoSearchModule,
                                         ZoneBasedTopoSearchModule)
from .modules.topology_nn_policy import TopoNNTopKModule
from .utils import ASSETS

logger = logging.getLogger(__name__)


class LJNAgent(BaseAgent):
    def __init__(
        self,
        action_space: ActionSpace,
        env,
        rho_danger: float = 0.99,
        rho_safe: float = 0.9,
        action_space_unsafe: str = os.path.join(ASSETS, "action_12_unsafe.npz"),
        action_space_n1_unsafe: str = os.path.join(ASSETS, "action_N1_unsafe.npz"),
        is_zone_based_topo: bool = False,
    ):
        BaseAgent.__init__(self, action_space=action_space)
        # Environment
        self.env = env
        self.rho_danger = rho_danger
        self.rho_safe = rho_safe
        # Sub-modules
        # Heuristic
        self.reconnect = RecoPowerlinePerAreaModule(
            self.action_space,
            env._game_rules.legal_action.substations_id_by_area,
            env._game_rules.legal_action.lines_id_by_area,
        )
        self.recover_topo = RecoverInitTopoModule(self.action_space)
        # Search topo
        if is_zone_based_topo:
            self.topo_12_unsafe = ZoneBasedTopoSearchModule(
                self.action_space,
                action_space_unsafe,
                self.env._game_rules.legal_action.substations_id_by_area,
                self.env.line_or_to_subid,
            )
            self.topo_n1_unsafe = ZoneBasedTopoSearchModule(
                self.action_space,
                action_space_n1_unsafe,
                self.env._game_rules.legal_action.substations_id_by_area,
                self.env.line_or_to_subid,
            )
        else:
            self.topo_12_unsafe = TopoSearchModule(
                self.action_space, action_space_unsafe
            )
            self.topo_n1_unsafe = TopoSearchModule(
                self.action_space, action_space_n1_unsafe
            )
        # Continuous control
        self.optim = OptimModule(env, self.action_space)

    def act(
        self, observation: BaseObservation, reward: float, done: bool = False
    ) -> BaseAction:
        start = time.time()

        # Init action with "do nothing"
        act = self.action_space()

        # Try to perform reconnection if necessary
        reco_act = self.reconnect.get_act(observation, act, reward)
        if reco_act is not None:
            _obs, _rew, _done, _info = observation.simulate(reco_act, time_step=1)
            if (
                reco_act is not None
                and not _done
                and reco_act != self.action_space({})
                and 0.0 < _obs.rho.max() < 2.0
                and (len(_info["exception"]) == 0)
            ):
                change = _obs.rho.max() - observation.rho.max()
                act += reco_act

        if observation.rho.max() > self.rho_danger:

            # Try to perform a topology recovery at first (= reconnection to bus 1)
            recovery_act = self.recover_topo.get_act(
                observation, act, reward, rho_threshold=self.rho_danger
            )
            if recovery_act is not None:
                act += recovery_act

            else:
                # Case 1 : All lines are connected
                if all(observation.line_status):
                    logger.warning("calling 12 unsafe")
                    topo_act = self.topo_12_unsafe.get_act(observation, act, reward)
                # Case 2 : N-1 situation, at least one line is disconnected
                else:

                    topo_act = self.topo_n1_unsafe.get_act(observation, act, reward)

                if topo_act is not None:
                    act += topo_act

            _obs, _rew, _done, _info = observation.simulate(act, time_step=1)
            if _obs.rho.max() > self.rho_safe or (len(_info["exception"]) != 0):
                # The problem has not been solved only by topology reconfiguration
                # Call the continuous control optimization module
                act = self.optim.get_act(observation, act, reward)

        elif _obs.rho.max() < self.rho_safe:
            # Try to find a recovery action when the grid is safe
            recovery_act = self.recover_topo.get_act(
                observation, act, reward, rho_threshold=0.8
            )
            if recovery_act is not None:
                act += recovery_act
        else:
            # Update the observed storage power.
            self.optim._update_storage_power_obs(observation)
            self.optim.flow_computed[:] = observation.p_or

        return act


class LJNAgentTopoNN(BaseAgent):
    def __init__(
        self,
        action_space: ActionSpace,
        env,
        gym_env,
        rho_danger: float = 0.99,
        rho_safe: float = 0.9,
        topk: int = 10,
        action_space_unsafe: str = os.path.join(ASSETS, "action_12_unsafe.npz"),
        action_space_n1_unsafe: str = os.path.join(ASSETS, "action_N1_unsafe.npz"),
        model_path: str = os.path.join(
            os.path.dirname(__file__), "models/baseline_12_unsafe_model.zip"
        ),
        training_mode: bool = False,
    ):
        BaseAgent.__init__(self, action_space=action_space)
        # Environment
        self.env = env
        self.rho_danger = rho_danger
        self.rho_safe = rho_safe
        # Sub-modules
        # Heuristic
        self.reconnect = RecoPowerlinePerAreaModule(
            self.action_space,
            env._game_rules.legal_action.substations_id_by_area,
            env._game_rules.legal_action.lines_id_by_area,
        )
        self.recover_topo = RecoverInitTopoModule(self.action_space)
        # Search topo
        self.topo_12_unsafe = TopoNNTopKModule(
            self.action_space,
            gym_env,
            model_path=model_path,
            top_k=topk,
        )
        self.topo_n1_unsafe = TopoSearchModule(
            self.action_space, action_space_n1_unsafe
        )
        # Continuous control
        self.optim = OptimModule(env, self.action_space)

        self.training_mode = training_mode

    def act(
        self, observation: BaseObservation, reward: float, done: bool = False
    ) -> BaseAction:
        start = time.time()

        # Init action with "do nothing"
        act = self.action_space()

        # Try to perform reconnection if necessary
        reco_act = self.reconnect.get_act(observation, act, reward)
        _obs, _rew, _done, _info = observation.simulate(reco_act, time_step=1)
        if (
            reco_act is not None
            and not _done
            and reco_act != self.action_space({})
            and 0.0 < _obs.rho.max() < 2.0
            and (len(_info["exception"]) == 0)
        ):
            change = _obs.rho.max() - observation.rho.max()
            act += reco_act

        if observation.rho.max() > self.rho_danger:

            # Try to perform a topology recovery at first (= reconnection to bus 1)
            recovery_act = self.recover_topo.get_act(
                observation, act, reward, rho_threshold=self.rho_danger
            )
            if recovery_act is not None:
                act += recovery_act

            else:
                # Case 1 : All lines are connected
                if all(observation.line_status):
                    if self.training_mode:
                        return None
                    topo_act = self.topo_12_unsafe.get_act(observation, act, reward)
                    if topo_act is not None:
                        _obs, _, _, _ = observation.simulate(act + topo_act)
                        rho_change = _obs.rho.max() - observation.rho.max()
                # Case 2 : N-1 situation, at least one line is disconnected
                else:
                    topo_act = self.topo_n1_unsafe.get_act(observation, act, reward)

                if topo_act is not None:
                    act += topo_act

            _obs, _rew, _done, _info = observation.simulate(act, time_step=1)
            if _obs.rho.max() > self.rho_safe or (len(_info["exception"]) != 0):
                # The problem has not been solved only by topology reconfiguration
                # Call the continuous control optimization module
                act = self.optim.get_act(observation, act, reward)

        elif _obs.rho.max() < self.rho_safe:
            # Try to find a recovery action when the grid is safe
            recovery_act = self.recover_topo.get_act(
                observation, act, reward, rho_threshold=0.8
            )
            if recovery_act is not None:
                act += recovery_act
        else:
            # Update the observed storage power.
            self.optim._update_storage_power_obs(observation)
            self.optim.flow_computed[:] = observation.p_or
        return act
