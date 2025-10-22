# Copyright (c) 2023-2024 La Javaness (https://lajavaness.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of L2RPN 2023 LJN Agent, a repository for the winning agent of L2RPN 2023 competition. It is a submodule contribution to the L2RPN Baselines repository.

import logging
import os

import numpy as np

logger = logging.getLogger(__name__)


def convert_from_vect(action_space, act):
    """
    Helper to convert an action, represented as a numpy array as an :class:`grid2op.BaseAction` instance.

    Parameters
    ----------
    act: ``numppy.ndarray``
        An action cast as an :class:`grid2op.BaseAction.BaseAction` instance.

    Returns
    -------
    res: :class:`grid2op.Action.Action`
        The `act` parameters converted into a proper :class:`grid2op.BaseAction.BaseAction` object.
    """
    res = action_space({})
    res.from_vect(act)
    return res


def load_action_to_grid2op(
    action_space, action_vec_path, action_threshold=None, return_counts=False
):
    """Load actions saved in npz file and convert them to grid2op action class instances.

    Args:
        action_space: obtain from env.action_space
        action_vec_path: a path ending with .npz. The loaded all_actions has two keys 'action_space', 'counts'.
            all_actions['action_space'] is a (N, K) matrix, where each row is an action.
        action_threshold: if provided, will filter out actions with counts below this.
        return_counts: if True, return the counts.
    """
    # load all_actions from npz file with two keys 'action_space', 'counts'
    assert action_vec_path.endswith(".npz") and os.path.exists(
        action_vec_path
    ), FileNotFoundError(
        f"file_paths {action_vec_path} does not contain a valid npz file."
    )
    data = np.load(action_vec_path, allow_pickle=True)
    all_actions = data["action_space"]
    assert isinstance(all_actions, np.ndarray), RuntimeError(
        f"Expect {action_vec_path} to be an ndarray, got {type(all_actions)}"
    )
    action_dim = action_space.size()
    assert all_actions.shape[1] == action_dim, RuntimeError(
        f"Expect {action_vec_path} to be an ndarray of shape {action_dim}, got {all_actions.shape[1]}"
    )
    counts = data["counts"]
    if action_threshold is not None:
        num_actions = (counts >= action_threshold).sum()
        all_actions = all_actions[:num_actions]
        counts = counts[:num_actions]
        logger.info("{} actions loaded.".format(num_actions))
    all_actions = [convert_from_vect(action_space, action) for action in all_actions]
    if return_counts:
        return all_actions, counts
    return all_actions
