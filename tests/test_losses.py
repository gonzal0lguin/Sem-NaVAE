"""Training loss terms (paper Eq. 3-5)."""

import pytest

torch = pytest.importorskip("torch")

from sem_navae.config import DataConfig, LossConfig  # noqa: E402
from sem_navae.losses import NaVAELoss  # noqa: E402

RES, HALF = 0.1, 300


def loss_fn(**kw):
    return NaVAELoss(LossConfig(**kw), DataConfig(semantic_map_resolution=RES))


def obstacle_map(x, y, size=0.5):
    """Square obstacle centred at (x, y) [m] with the dataset convention map[i <-> x, j <-> y]."""
    m = torch.zeros(1, 2 * HALF, 2 * HALF)
    i, j, r = int(x / RES + HALF), int(y / RES + HALF), int(size / RES / 2)
    m[0, i - r : i + r, j - r : j + r] = 1.0
    return m


def straight(y=0.0, n=12):
    xs = torch.arange(1, n + 1, dtype=torch.float)
    return torch.stack([xs, torch.full_like(xs, y)], -1).view(1, 1, n, 2)


def test_free_path_costs_nothing():
    assert loss_fn().collision_loss(straight(), obstacle_map(5.0, 8.0)).item() == pytest.approx(
        0.0, abs=1e-6
    )


def test_cost_decreases_with_distance_to_obstacle():
    m = obstacle_map(5.0, 0.0)
    costs = [loss_fn().collision_loss(straight(y), m).item() for y in (0.0, 0.3, 0.6, 2.0)]
    assert costs[0] > costs[1] > costs[2] > costs[3]


def test_axes_follow_the_dataset_convention():
    on_x_axis = obstacle_map(5.0, 0.0)  # in front of the robot
    on_y_axis = obstacle_map(0.0, 5.0)  # to its left
    loss = loss_fn().collision_loss
    assert loss(straight(), on_x_axis).item() > 0.1
    assert loss(straight(), on_y_axis).item() == pytest.approx(0.0, abs=1e-6)


def test_collision_loss_has_a_gradient():
    # T1: the legacy term was computed on detached positions and had no gradient.
    path = straight(y=0.2).requires_grad_(True)
    loss_fn().collision_loss(path, obstacle_map(5.0, 0.0)).sum().backward()
    grad = path.grad[0, 0]
    assert grad.abs().sum() > 0
    assert grad[4, 1] < 0  # descending moves the waypoint near x = 5 m away from y = 0


def test_output_shape():
    paths = torch.zeros(2, 3, 12, 2)
    maps = torch.cat([obstacle_map(1.0, 1.0)] * 2)
    assert loss_fn().collision_loss(paths, maps).shape == (2, 3)
