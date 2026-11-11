import numpy as np
import pytest

from sem_navae.selection import RigidTransform, project_to_image, quaternion_to_matrix

K = np.array([[100.0, 0.0, 50.0], [0.0, 100.0, 40.0], [0.0, 0.0, 1.0]])
SIZE = (100, 80)  # (width, height)


def test_quaternion_to_matrix_yaw():
    yaw = 0.7
    q = [0.0, 0.0, np.sin(yaw / 2), np.cos(yaw / 2)]
    np.testing.assert_allclose(
        quaternion_to_matrix(q), RigidTransform.from_xy_yaw(0, 0, yaw).rotation, atol=1e-12
    )


def test_transform_compose_and_inverse():
    a = RigidTransform.from_quaternion([1.0, 2.0, 0.5], [0.1, -0.2, 0.3, 0.9])
    b = RigidTransform.from_xy_yaw(-3.0, 0.5, 1.2)
    p = np.random.default_rng(0).normal(size=(4, 5, 3))
    np.testing.assert_allclose((a @ b).apply(p), a.apply(b.apply(p)))
    np.testing.assert_allclose(a.inverse().apply(a.apply(p)), p, atol=1e-12)


def test_body_frame_projection_center_and_sides():
    pts = np.array([[5.0, 0.0, 0.0], [5.0, 1.0, 0.0], [5.0, 0.0, 1.0]])
    px, valid = project_to_image(pts, K, SIZE)
    np.testing.assert_allclose(px[0], [50.0, 40.0])  # on the optical axis
    assert px[1, 0] < 50.0  # +y (left) appears left in the image
    assert px[2, 1] < 40.0  # +z (up) appears higher in the image
    assert valid.all()


def test_optical_frame_matches_body_frame():
    body = np.array([[4.0, 0.5, -0.3]])
    optical = np.array([[-0.5, 0.3, 4.0]])
    from_body, _ = project_to_image(body, K, SIZE)
    from_optical, _ = project_to_image(optical, K, SIZE, optical_frame=True)
    np.testing.assert_allclose(from_body, from_optical)


@pytest.mark.parametrize("depth", [-3.0, 0.0, 0.05])
def test_points_behind_or_at_camera_are_invalid(depth):
    px, valid = project_to_image(np.array([[depth, 0.0, 0.0]]), K, SIZE)
    assert not valid[0]
    assert np.isnan(px[0]).all()


def test_out_of_image_is_invalid_on_every_border():
    pts = np.array(
        [
            [1.0, 5.0, 0.0],  # far left
            [1.0, -5.0, 0.0],  # far right
            [1.0, 0.0, 5.0],  # above
            [1.0, 0.0, -5.0],  # below
        ]
    )
    _, valid = project_to_image(pts, K, SIZE)
    assert not valid.any()
