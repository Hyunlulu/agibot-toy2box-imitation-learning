# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Verify that the Agibot Toy2Box task exposes a usable Robomimic BC config."""

import importlib
import json
from pathlib import Path

import gymnasium as gym

from isaaclab.app import AppLauncher


app_launcher = AppLauncher(headless=True)
simulation_app = app_launcher.app

import isaaclab_tasks  # noqa: E402, F401


def test_agibot_toy2box_robomimic_bc_config_is_loadable():
    """The training entry point must resolve to the expected low-dimensional BC config."""
    spec = gym.spec("Isaac-Place-Toy2Box-Agibot-Right-Arm-RmpFlow-v0")
    entry_point = spec.kwargs["robomimic_bc_cfg_entry_point"]
    module_name, relative_path = entry_point.split(":", 1)
    module = importlib.import_module(module_name)
    config_path = Path(module.__file__).parent / relative_path

    with config_path.open(encoding="utf-8") as config_file:
        config = json.load(config_file)

    assert config["algo_name"] == "bc"
    assert config["experiment"]["name"] == "bc_rnn_low_dim_agibot_toy2box"
    assert config["observation"]["modalities"]["obs"]["low_dim"] == [
        "eef_pos",
        "eef_quat",
        "gripper_pos",
        "toy_truck_positions",
        "toy_truck_orientations",
        "box_positions",
        "box_orientations",
    ]
