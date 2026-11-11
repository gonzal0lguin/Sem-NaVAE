import numpy as np
import pytest

from sem_navae.selection import (
    SemanticCostmap,
    build_prompts,
    discount,
    goal_cost,
    load_selection_config,
    resize_nearest,
    sample_waypoint_costs,
    semantic_cost,
)
from sem_navae.utils.geometry import get_path_dist_to_goal


def _straight(direction, n=5):
    d = np.asarray(direction, dtype=float) / np.linalg.norm(direction)
    return (np.arange(1, n + 1)[:, None] * d)[None]


def test_heading_towards_goal_scores_zero():  # R4
    goal = np.array([20.0, 0.0])
    towards = get_path_dist_to_goal(_straight([1, 0]), goal)[0, 1]
    away = get_path_dist_to_goal(_straight([-1, 0]), goal)[0, 1]
    assert towards == pytest.approx(0.0)
    assert abs(away) == pytest.approx(np.pi)


def test_goal_cost_prefers_paths_towards_goal():
    goal = np.array([10.0, 10.0])
    paths = np.concatenate([_straight([1, 1]), _straight([1, -1]), _straight([-1, -1])])
    c = goal_cost(paths, goal, alpha1=2.0, alpha2=0.2)
    assert c[0] < c[1] < c[2]
    rho = np.linalg.norm(paths[0, -1] - goal)
    assert c[0] == pytest.approx(2.0 * np.log1p(rho))


def test_eq7_argmax_then_cost_and_nearest_resize():
    scores = np.zeros((3, 2, 2))
    scores[0, 0, 0] = scores[1, 0, 1] = scores[2, 1, :] = 1.0
    cm = SemanticCostmap.from_scores(scores, [0.0, 3.0, 2.0], image_size=(4, 4))
    expected = np.array([[0, 0, 3, 3], [0, 0, 3, 3], [2, 2, 2, 2], [2, 2, 2, 2]], dtype=float)
    np.testing.assert_array_equal(cm.costs, expected)
    assert set(np.unique(cm.costs)) <= {0.0, 2.0, 3.0}  # never blended


def test_resize_nearest_never_creates_new_values():
    img = np.random.default_rng(0).integers(0, 5, size=(7, 11))
    out = resize_nearest(img, (23, 5))
    assert out.shape == (23, 5)
    assert set(np.unique(out)) <= set(np.unique(img))


def test_prompt_template():
    assert build_prompts(["grass", "tree"]) == ["a photo of a grass", "a photo of a tree"]


def test_waypoint_costs_occlusion_and_invalid_are_unknown():
    cost_image = np.array([[0.0, 1.0, 2.0, 3.0]])
    pixels = np.array([[[0.2, 0.0], [1.5, 0.0], [2.0, 0.0], [3.9, 0.0], [np.nan, np.nan]]])
    valid = np.array([[True, True, True, True, False]])
    costs, observed = sample_waypoint_costs(cost_image, pixels, valid, 2.5, occlusion_threshold=2)
    np.testing.assert_array_equal(costs, [[0.0, 1.0, 2.0, 2.5, 2.5]])
    np.testing.assert_array_equal(observed, valid)


def test_eq8_discount_starts_at_one():
    np.testing.assert_allclose(discount(3, 0.5), [0.5, 0.25, 0.125])
    assert semantic_cost(np.array([1.0, 1.0, 1.0]), 0.5) == pytest.approx(0.875)


def test_default_selection_config_matches_paper(tmp_path):
    import os

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cfg = load_selection_config(os.path.join(root, "configs", "selection", "default.yaml"))
    assert (cfg.gamma, cfg.alpha1, cfg.alpha2) == (0.8, 2.0, 0.2)
    assert cfg.unknown_cost == cfg.occlusion_threshold == 2.0
    assert cfg.classes["sky"] == 4 and cfg.classes["pavement"] == 0
    assert cfg.mode == "cost_to_go"
    with pytest.raises(KeyError):
        from sem_navae.selection import selection_config_from_dict

        selection_config_from_dict({"gamma_typo": 1})
