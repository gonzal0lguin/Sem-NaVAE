"""The heatmap builder and the decoder agree on where a metric point lands (T6)."""

import pytest

torch = pytest.importorskip("torch")

from sem_navae.config import HeatmapSpec  # noqa: E402
from sem_navae.models.heatmap import metric_to_pixel, traversability_heatmap  # noqa: E402

NORM = 50.0


def blob(center, n=200, spread=0.15, seed=0):
    g = torch.Generator().manual_seed(seed)
    xy = torch.tensor(center) + spread * torch.randn(n, 2, generator=g)
    return torch.cat([xy, torch.zeros(n, 2)], dim=-1)


def build(spec, free_at, obstacle_at):
    pts = torch.cat([blob(free_at), blob(obstacle_at, seed=1)]).unsqueeze(0)
    labels = torch.cat([torch.zeros(200), torch.ones(200)]).long().unsqueeze(0)
    probs = torch.stack([1.0 - labels.float(), labels.float()], -1) * 0.98 + 0.01
    pts = pts.clone()
    pts[..., :3] /= NORM
    return traversability_heatmap(pts, probs.log(), labels, spec, NORM)[0]


@pytest.mark.parametrize("spec", [HeatmapSpec(), HeatmapSpec(size_m=30.0, resolution=0.1)])
def test_peaks_are_where_the_points_are(spec):
    free_at, obstacle_at = (3.0, -2.0), (-4.0, 5.0)
    hm = build(spec, free_at, obstacle_at)
    assert hm.shape == (spec.size_px, spec.size_px)
    for target, value in [(free_at, hm.max()), (obstacle_at, hm.min())]:
        col, row = metric_to_pixel(torch.tensor(target), spec).tolist()
        assert hm[row, col] == pytest.approx(value.item(), abs=0.05)
        rows, cols = torch.nonzero(hm == value, as_tuple=True)
        assert abs(rows[0].item() - row) <= 2 and abs(cols[0].item() - col) <= 2
    assert hm[metric_to_pixel(torch.tensor(free_at), spec).flip(0).unbind()] > 0
    assert hm[metric_to_pixel(torch.tensor(obstacle_at), spec).flip(0).unbind()] < 0


def test_robot_is_at_the_centre():
    spec = HeatmapSpec()
    assert metric_to_pixel(torch.zeros(2), spec).tolist() == [spec.size_px // 2] * 2
    assert spec.size_px == 90  # paper: 18 m x 18 m at 0.2 m/px
