"""Record the legacy selector cost (``_calculate_path_cost_full``) for the parity tests.

Usage:
    python tests/golden/generate_selection_from_legacy.py /path/to/NaVAE

ROS modules are replaced by stubs and tf2 by a fixed lookup table, so no ROS install is
needed. Writes ``tests/golden/legacy_selection.npz``.
"""

from __future__ import annotations

import os
import sys
from types import SimpleNamespace
from unittest import mock

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import selection_fixtures as fx  # noqa: E402

ROS_MODULES = [
    "rospy",
    "tf",
    "tf.transformations",
    "actionlib",
    "move_base_msgs",
    "move_base_msgs.msg",
    "sensor_msgs",
    "sensor_msgs.msg",
    "geometry_msgs",
    "geometry_msgs.msg",
    "std_msgs",
    "std_msgs.msg",
    "nav_msgs",
    "nav_msgs.msg",
    "navae_ros",
    "navae_ros.msg",
    "rl_navigation",
    "rl_navigation.msg",
]


class FakeBuffer:
    def lookup_transform(self, target, source, *_args, **_kwargs):
        t, q = fx.TF[(target, source)]
        return SimpleNamespace(
            transform=SimpleNamespace(
                translation=SimpleNamespace(x=t[0], y=t[1], z=t[2]),
                rotation=SimpleNamespace(x=q[0], y=q[1], z=q[2], w=q[3]),
            )
        )


def main(legacy_root: str) -> None:
    for name in ROS_MODULES:
        sys.modules[name] = mock.MagicMock()
    tf2 = mock.MagicMock()
    tf2.LookupException = type("LookupException", (Exception,), {})
    sys.modules["tf2_ros"] = tf2
    sys.path[:0] = [legacy_root, os.path.join(legacy_root, "ros", "navae_ros", "scripts")]
    import selector  # legacy ros/navae_ros/scripts/selector.py

    sel = selector.TrajectorySelector.__new__(selector.TrajectorySelector)
    sel.tf_buffer = FakeBuffer()
    sel._K = fx.INTRINSICS
    sel._gamma = fx.GAMMA
    sel.C_unk = fx.UNKNOWN_COST
    sel._beta = fx.ALPHA1
    sel._beta2 = fx.ALPHA2
    sel._ignore_semantics = False
    sel.robot_frame = "base_link"
    sel.global_frame = "map"

    costs, wpt_costs, (pixels, outmask) = sel._calculate_path_cost_full(
        fx.costmap(), fx.paths(), fx.GOAL_MAP, source_frame="base_link"
    )
    out = dict(path_costs=costs, waypoint_costs=wpt_costs, pixels=pixels, outmask=outmask)
    np.savez_compressed(os.path.join(HERE, "legacy_selection.npz"), **out)
    print("wrote", os.path.join(HERE, "legacy_selection.npz"))


if __name__ == "__main__":
    main(sys.argv[1])
