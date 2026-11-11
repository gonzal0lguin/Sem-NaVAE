"""Parity of sem_navae.selection with the legacy selector's ``_calculate_path_cost_full``.

Golden values come from ``tests/golden/generate_selection_from_legacy.py``. The port fixes
R4 (goal heading), R6 (projection validity) and uses Eq. 8's j = 1..N discount, so the
checks below state each relation to the legacy numbers explicitly.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "golden"))
import selection_fixtures as fx  # noqa: E402

from sem_navae.selection import (  # noqa: E402
    RigidTransform,
    goal_cost,
    project_to_image,
    sample_waypoint_costs,
    semantic_cost,
)

GOLDEN = np.load(os.path.join(os.path.dirname(__file__), "golden", "legacy_selection.npz"))


def _tf(target, source):
    return RigidTransform.from_quaternion(*fx.TF[(target, source)])


def _project():
    pts = _tf("camera_link", "base_link").apply(fx.paths())
    return project_to_image(pts, fx.INTRINSICS, (fx.IMG_W, fx.IMG_H))


def test_projection_matches_legacy_in_front_of_camera():
    pixels, valid = _project()
    depth = _tf("camera_link", "base_link").apply(fx.paths())[..., 0]
    front = depth > 0.1
    np.testing.assert_allclose(pixels[front], GOLDEN["pixels"][front])
    # R6: the backwards path (index 4) is behind the camera; the legacy code mirrored it
    # into the image (top rows) instead of rejecting it.
    assert not valid[4].any()
    legacy = GOLDEN["pixels"][4]
    assert ((legacy[:, 0] >= 0) & (legacy[:, 0] < fx.IMG_W) & (legacy[:, 1] >= 0)).all()


def test_semantic_cost_relation_to_legacy():
    pixels, valid = _project()
    costs, observed = sample_waypoint_costs(
        fx.costmap(), pixels, valid, fx.UNKNOWN_COST, occlusion_threshold=1.0
    )
    # Observed waypoints sample the same pixels as the legacy code (legacy T_occ was a
    # hard-coded "> 1"; legacy stored gamma-weighted costs).
    legacy_unweighted = GOLDEN["waypoint_costs"] / fx.GAMMA ** np.arange(12)
    assert observed.sum() >= 20  # the relation is actually exercised
    np.testing.assert_allclose(costs[observed], legacy_unweighted[observed])
    # Eq. 8 discounts from j = 1; the legacy code started at gamma^0.
    np.testing.assert_allclose(
        semantic_cost(legacy_unweighted, fx.GAMMA), fx.GAMMA * GOLDEN["waypoint_costs"].sum(1)
    )
    # Unobserved waypoints are unknown (C_u), not the clipped border pixel.
    assert (costs[~observed] == fx.UNKNOWN_COST).all()


def test_goal_cost_relation_to_legacy():
    tf_bm = _tf("base_link", "map")
    goal_base = tf_bm.apply(np.append(fx.GOAL_MAP, 0.0))[:2]
    paths = fx.paths()
    new = goal_cost(paths, goal_base, fx.ALPHA1, fx.ALPHA2)
    legacy_sem = GOLDEN["waypoint_costs"].sum(axis=1)
    legacy_goal = GOLDEN["path_costs"] - legacy_sem
    dist_term = fx.ALPHA1 * np.log1p(np.linalg.norm(paths[:, -1, :2] - goal_base, axis=1))
    # R4: heading term flips, alpha2*|theta|/pi -> alpha2*(1 - |theta|/pi).
    np.testing.assert_allclose(new - dist_term, fx.ALPHA2 - (legacy_goal - dist_term))
