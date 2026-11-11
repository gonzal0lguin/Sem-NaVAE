"""Replay a short synthetic sequence through the ROS-free SelectionCore and check decisions."""

import numpy as np
import pytest

from sem_navae.selection import RigidTransform, SelectionConfig, SelectionCore

K = np.array([[100.0, 0.0, 50.0], [0.0, 100.0, 40.0], [0.0, 0.0, 1.0]])
W, H = 100, 80
T_CAM_FROM_ROBOT = RigidTransform(np.eye(3), np.array([0.0, 0.0, -0.6]))  # camera 0.6 m up
GOAL = np.array([30.0, 0.0])
X = np.arange(2.0, 8.0)
LEFT = np.stack([X, np.full_like(X, 0.8), np.zeros_like(X)], -1)  # projects to u in [10, 40]
RIGHT = LEFT * np.array([1.0, -1.0, 1.0])  # u in [60, 90]
PATHS = np.stack([LEFT, RIGHT])


def _costmap(left, right):
    cm = np.zeros((H, W))
    cm[:, : W // 2], cm[:, W // 2 :] = left, right
    return cm


def _core(mode="cost_to_go"):
    return SelectionCore(SelectionConfig(mode=mode), K, (W, H))


def test_replay_switching_sequence():
    core = _core()
    pose_a = RigidTransform.from_xy_yaw(10.0, 5.0, np.pi / 2)  # robot in the fixed frame
    cam_from_fixed = T_CAM_FROM_ROBOT @ pose_a.inverse()

    # t0: grass (2) on the left, pavement (0) on the right -> take the right path.
    core.on_costmap(_costmap(2, 0), cam_from_fixed)
    d, fixed = core.on_candidates(PATHS, GOAL, T_CAM_FROM_ROBOT, pose_a)
    assert d.switch and d.index == 1
    np.testing.assert_allclose(fixed, pose_a.apply(RIGHT))

    # t1: a person (3 > T_occ) appears on the right; y_curr is re-scored to unknown cost.
    j_before = core.policy.current_cost()
    j_after = core.on_costmap(_costmap(0, 3), cam_from_fixed)
    assert j_after > j_before

    # t2: a new candidate set; the left path now wins by more than epsilon -> switch.
    d, _ = core.on_candidates(PATHS, GOAL, T_CAM_FROM_ROBOT, pose_a)
    assert d.switch and d.index == 0

    # t3: an almost identical set; within the hysteresis -> keep y_curr.
    d, fixed = core.on_candidates(PATHS + [0.0, 0.01, 0.0], GOAL, T_CAM_FROM_ROBOT, pose_a)
    assert not d.switch and fixed is None
    np.testing.assert_allclose(core.current_waypoints, pose_a.apply(LEFT))


def test_stored_trajectory_does_not_drift_when_robot_moves():
    core = _core()
    pose_a = RigidTransform.from_xy_yaw(1.0, 2.0, 0.3)
    core.on_costmap(_costmap(0, 0))
    core.on_candidates(PATHS, GOAL, T_CAM_FROM_ROBOT, pose_a)
    stored = core.current_waypoints.copy()
    pose_b = RigidTransform.from_xy_yaw(3.0, 2.5, 0.5)  # the robot moved
    core.on_costmap(_costmap(0, 0), T_CAM_FROM_ROBOT @ pose_b.inverse())
    np.testing.assert_array_equal(core.current_waypoints, stored)


def test_infeasible_paths_are_skipped_and_index_maps_back():
    core = _core()
    core.on_costmap(_costmap(2, 0))
    d, _ = core.on_candidates(
        PATHS, GOAL, T_CAM_FROM_ROBOT, RigidTransform.identity(), feasible=np.array([True, False])
    )
    assert d.switch and d.index == 0


def test_periodic_mode_switches_every_set():
    core = _core("periodic")
    core.on_costmap(_costmap(2, 0))
    for _ in range(3):
        d, _ = core.on_candidates(PATHS, GOAL, T_CAM_FROM_ROBOT, RigidTransform.identity())
        assert d.switch and d.index == 1


def test_candidates_before_costmap_raise():
    with pytest.raises(RuntimeError):
        _core().on_candidates(PATHS, GOAL, T_CAM_FROM_ROBOT, RigidTransform.identity())
