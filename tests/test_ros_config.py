"""The ROS parameter file stays in sync with the ROS-free selection config."""

import os

import yaml

from sem_navae.selection import load_selection_config, selection_config_from_dict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_ros_selection_block_matches_default_config():
    with open(os.path.join(ROOT, "ros", "sem_navae_ros", "config", "sem_navae.yaml")) as f:
        ros = yaml.safe_load(f)
    default = load_selection_config(os.path.join(ROOT, "configs", "selection", "default.yaml"))
    assert selection_config_from_dict(ros["selection"]) == default
    assert ros["model"]["trajectory_gen_period"] == 2.0  # f_gen = 0.5 Hz
