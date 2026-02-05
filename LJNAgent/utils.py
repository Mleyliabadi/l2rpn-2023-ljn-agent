# Copyright (c) 2023-2024 La Javaness (https://lajavaness.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of L2RPN 2023 LJN Agent, a repository for the winning agent of L2RPN 2023 competition. It is a submodule contribution to the L2RPN Baselines repository.

import os

CHALLENGE_ENV = "l2rpn_idf_2023"
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
ASSETS = os.path.join(os.path.dirname(__file__), "assets")

NN_ACT_SPACE_DIR = os.path.join(os.path.dirname(__file__), "assets/nn_act_space/")

DEFAULT_OBS_ATTR_TO_KEEP = [
    "day_of_week",
    "hour_of_day",
    "minute_of_hour",
    "prod_p",
    "prod_v",
    "load_p",
    "load_q",
    "actual_dispatch",
    "target_dispatch",
    "topo_vect",
    "time_before_cooldown_line",
    "time_before_cooldown_sub",
    "rho",
    "timestep_overflow",
    "line_status",
    "storage_power",
    "storage_charge",
]
