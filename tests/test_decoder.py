"""Rejection-sampling constraints of the decoder (paper Sec. III-A3)."""

import math

import pytest

torch = pytest.importorskip("torch")

from sem_navae.config import HeatmapSpec  # noqa: E402
from sem_navae.models.decoder import StochasticDecoder  # noqa: E402


def decoder(**kw):
    args = dict(
        context_dim=8,
        latent_dim=8,
        steps=3,
        heatmap=HeatmapSpec(),
        use_heatmap=False,
        use_map_features=False,
        max_accel=10.0,
        max_heading_change=math.radians(45),
    )
    args.update(kw)
    return StochasticDecoder(**args)


def pick(dec, proposals, prev, heatmap=None, preds=None):
    """Proposals ``(K, 2)`` for a single batch element; returns the chosen one."""
    prev = torch.tensor([prev])
    mu = torch.full((1, 2), 99.0)  # returned if nothing is valid
    out = dec.constrained_sample(
        torch.tensor(proposals).unsqueeze(1), prev, heatmap, mu, preds or [prev]
    )
    return out[0].tolist()


def rot(deg, speed=1.0):
    return [speed * math.cos(math.radians(deg)), speed * math.sin(math.radians(deg))]


def test_heading_is_the_angle_between_velocities():
    dec = decoder()
    assert pick(dec, [rot(60), rot(30)], prev=[1.0, 0.0]) == pytest.approx(rot(30))
    assert pick(dec, [rot(-50), rot(170)], prev=[1.0, 0.0]) == [99.0, 99.0]
    # straight motion at constant heading is always allowed, whatever happened before
    assert pick(dec, [[1.2, 0.0]], prev=[1.0, 0.0]) == pytest.approx([1.2, 0.0])


def test_heading_is_free_when_stopped():
    assert pick(decoder(), [rot(120, 0.5)], prev=[0.0, 0.0]) == pytest.approx(rot(120, 0.5))


def test_acceleration_limit_uses_dt():
    dec = decoder(max_accel=0.5)
    assert pick(dec, [[1.6, 0.0], [1.4, 0.0]], prev=[1.0, 0.0]) == pytest.approx([1.4, 0.0])
    slow = decoder(max_accel=0.5, dt=2.0)  # 0.6 m/s change over 2 s = 0.3 m/s^2
    assert pick(slow, [[1.6, 0.0]], prev=[1.0, 0.0]) == pytest.approx([1.6, 0.0])


def test_speed_limits():
    dec = decoder(min_speed=0.1, max_speed=2.0)
    assert pick(dec, [[0.05, 0.0], [2.5, 0.0], [1.5, 0.0]], prev=[1.0, 0.0]) == pytest.approx(
        [1.5, 0.0]
    )


def test_heatmap_rejects_waypoints_on_blocked_cells():
    spec = HeatmapSpec()
    dec = decoder(heatmap=spec, use_heatmap=True)
    hm = torch.ones(1, spec.size_px, spec.size_px)
    c = spec.size_px // 2
    hm[0, c, c + 7 : c + 9] = -1.0  # x in [1.4, 1.8) m, y = 0: blocked
    prev = [1.0, 0.0]
    preds = [torch.tensor([prev]), torch.tensor([[0.5, 0.0]])]  # already at x = 0.5 m
    out = pick(dec, [[1.1, 0.0], [0.8, 0.0]], prev=prev, heatmap=hm, preds=preds)
    assert out == pytest.approx([0.8, 0.0])  # 0.5 + 1.1 = 1.6 m is blocked, 1.3 m is free


def test_sampling_without_heatmap_feeds_back_samples():
    torch.manual_seed(0)
    dec = decoder(num_proposals=64)
    out = dec(torch.zeros(4, 8), torch.randn(4, 8), current_state=torch.tensor([[1.0, 0.0]] * 4))
    assert out.mean.shape == (4, 3, 2)
