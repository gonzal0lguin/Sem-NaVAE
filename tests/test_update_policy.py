"""Properties of the asynchronous update policy (Sec. III-B4)."""

import numpy as np
import pytest

from sem_navae.selection import UpdatePolicy, semantic_cost

N = 6
RNG = np.random.default_rng(0)


def _adopt(policy, wpt_costs, goal=1.0):
    d = policy.on_candidates(np.asarray([wpt_costs], dtype=float), np.array([goal]))
    assert d.switch
    return d


def test_first_candidate_set_is_always_adopted():
    p = UpdatePolicy(0.8, 0.15)
    d = p.on_candidates(np.array([[3.0] * N, [0.0] * N]), np.array([0.0, 0.0]))
    assert d.switch and d.index == 1


@pytest.mark.parametrize("mode", ["cost_to_go", "paper"])
def test_no_switch_within_hysteresis(mode):
    eps = 0.5
    for _ in range(50):
        p = UpdatePolicy(0.8, eps, mode)
        cur = RNG.uniform(0, 3, N)
        _adopt(p, cur, goal=1.0)
        j_curr = p.current_cost()
        new = RNG.uniform(0, 3, (4, N))
        goals = RNG.uniform(0, 2, 4)
        j_new = p.candidate_costs(new, goals)
        d = p.on_candidates(new, goals)
        assert d.switch == (j_new.min() < j_curr - eps)
        if not d.switch:
            np.testing.assert_array_equal(p.current.waypoint_costs, cur)


def test_periodic_always_switches_even_to_worse():
    p = UpdatePolicy(0.8, 0.15, "periodic")
    _adopt(p, [0.0] * N, goal=0.0)
    d = p.on_candidates(np.array([[3.0] * N]), np.array([5.0]))
    assert d.switch and d.index == 0


@pytest.mark.parametrize("mode", ["cost_to_go", "paper"])
def test_frozen_costs_never_change(mode):
    p = UpdatePolicy(0.8, 0.15, mode)
    _adopt(p, RNG.uniform(0, 3, N))
    for step in range(N + 1):
        p.on_progress(step)
        frozen = p.current.waypoint_costs[:step].copy()
        p.on_costmap(RNG.uniform(0, 3, N), np.ones(N, bool))
        np.testing.assert_array_equal(p.current.waypoint_costs[:step], frozen)


def test_progress_is_monotone():
    p = UpdatePolicy(0.8, 0.15)
    _adopt(p, [1.0] * N)
    p.on_progress(4)
    p.on_progress(2)
    assert p.current.progress == 4


def test_unobserved_future_waypoints_keep_previous_cost():
    p = UpdatePolicy(0.8, 0.15)
    _adopt(p, [1.0] * N)
    observed = np.array([False, True, False, True, True, False])
    p.on_costmap(np.full(N, 2.0), observed)
    np.testing.assert_array_equal(p.current.waypoint_costs, np.where(observed, 2.0, 1.0))


@pytest.mark.parametrize("mode", ["cost_to_go", "paper"])
def test_cost_is_monotone_in_occlusions(mode):
    """With C_u >= T_occ, turning observed waypoints into unknown never lowers J."""
    c_u, t_occ = 2.0, 2.0
    for _ in range(50):
        p = UpdatePolicy(0.8, 0.15, mode)
        _adopt(p, RNG.uniform(0, t_occ, N))
        p.on_progress(int(RNG.integers(0, N)))
        base = RNG.uniform(0, t_occ, N)
        p.on_costmap(base, np.ones(N, bool))
        j0 = p.current_cost()
        occl = RNG.random(N) < 0.5
        p.on_costmap(np.where(occl, c_u, base), np.ones(N, bool))
        assert p.current_cost() >= j0 - 1e-12


def test_cost_to_go_vs_paper_bookkeeping():
    costs = np.array([3.0, 3.0, 0.0, 0.0, 0.0, 0.0])
    a, b = UpdatePolicy(0.5, 0.0, "cost_to_go"), UpdatePolicy(0.5, 0.0, "paper")
    for p in (a, b):
        _adopt(p, costs, goal=0.0)
        p.on_progress(2)
    # cost_to_go: remaining waypoints only, re-indexed from the robot (all zero here).
    assert a.current_cost() == pytest.approx(0.0)
    # paper: frozen traversed costs, original indices.
    assert b.current_cost() == pytest.approx(semantic_cost(costs, 0.5))


def test_finished_trajectory_is_replaced():
    p = UpdatePolicy(0.8, 10.0)
    _adopt(p, [0.0] * N, goal=0.0)
    p.on_progress(N)
    d = p.on_candidates(np.array([[3.0] * N]), np.array([5.0]))
    assert d.switch


def test_invalid_mode():
    with pytest.raises(ValueError):
        UpdatePolicy(0.8, 0.1, "barebones")
