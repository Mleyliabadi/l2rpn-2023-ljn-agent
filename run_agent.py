# Copyright (c) 2023-2024 La Javaness (https://lajavaness.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of L2RPN 2023 LJN Agent, a repository for the winning agent of L2RPN 2023 competition. It is a submodule contribution to the L2RPN Baselines repository.

import logging
import re
from multiprocessing import Pool

import grid2op
from grid2op.Chronics import MultifolderWithCache
from grid2op.Runner import Runner
from lightsim2grid.lightSimBackend import LightSimBackend
from tqdm import tqdm

from .LJNagent import LJNAgent


def run_expert_agent(id: int):
    """This function use the grid2op runner on a single core with the training env

    Parameters
    ----------
    id : int
        index for multi-processing purpose
    """
    env = grid2op.make(
        "l2rpn_idf_2023", backend=LightSimBackend(), chronics_class=MultifolderWithCache
    )
    env.chronics_handler.real_data.set_filter(
        lambda x: re.match(f".*_{id}$", x) is not None
    )
    env.chronics_handler.real_data.reset()

    NB_EPISODE = 52  # len(os.listdir('data/l2rpn_idf_2023_train/chronics'))
    NB_CORE = 1
    logging.info(f"Starting runner on {NB_EPISODE} episodes on {NB_CORE}")
    PATH_SAVE = f"agents_log_BC/"  # and store the results in the "agents_log" folder

    # initialize agent
    agent = LJNAgent(env, env.action_space)
    agent.id_filter = id
    # use the runner
    runner = Runner(**env.get_params_for_runner(), agentClass=None, agentInstance=agent)
    runner.run(nb_episode=NB_EPISODE, nb_process=NB_CORE, path_save=PATH_SAVE)
    agent.reset(env.reset())
    logging.info("Done")


if __name__ == "__main__":
    ## Using multi-processing here instead of multiple processes with the runner because of a bug ##
    pool = Pool(processes=16)
    pool.map(run_expert_agent, range(16))
