"""The local planner plugin loader and the ROS-free built-ins (no ROS needed)."""

import sys
import textwrap
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ros" / "sem_navae_ros" / "src"))

from sem_navae_ros.local_planners import (  # noqa: E402
    BUILTIN_PLANNERS,
    LocalPlanner,
    PlannerStatus,
    PluginError,
    load_local_planner,
)
from sem_navae_ros.local_planners.null import NullPlanner  # noqa: E402

DUMMY_PLUGIN = textwrap.dedent(
    """
    from sem_navae_ros.local_planners import LocalPlanner, PlannerStatus

    class DummyPlanner(LocalPlanner):
        waypoint_stride = 3

        def __init__(self, params):
            super().__init__(params)
            self.sent = []
            self._status = PlannerStatus.IDLE

        def send_waypoint(self, xy, frame_id, yaw=None):
            self.sent.append((tuple(xy), frame_id, yaw))
            self._status = PlannerStatus.ACTIVE

        def cancel(self):
            self._status = PlannerStatus.IDLE

        def status(self):
            return self._status

    class NotAPlanner:
        pass

    class NeedsMissingDep(DummyPlanner):
        def __init__(self, params):
            super().__init__(params)
            import a_ros_package_that_is_not_installed  # noqa: F401
    """
)


@pytest.fixture
def plugin_module(tmp_path, monkeypatch):
    (tmp_path / "my_planners.py").write_text(DUMMY_PLUGIN)
    monkeypatch.syspath_prepend(str(tmp_path))
    yield "my_planners"
    sys.modules.pop("my_planners", None)


def test_load_plugin_from_string_path(plugin_module):
    planner = load_local_planner(f"{plugin_module}:DummyPlanner", {"foo": 1})
    assert isinstance(planner, LocalPlanner)
    assert planner.params == {"foo": 1}
    assert planner.waypoint_stride == 3  # class default
    assert planner.status() is PlannerStatus.IDLE
    planner.send_waypoint((1.0, 2.0), "odom")
    assert planner.sent == [((1.0, 2.0), "odom", None)]
    assert planner.status() is PlannerStatus.ACTIVE
    planner.cancel()
    assert planner.status() is PlannerStatus.IDLE


def test_waypoint_stride_param_overrides_class_default(plugin_module):
    planner = load_local_planner(f"{plugin_module}:DummyPlanner", {"waypoint_stride": 2})
    assert planner.waypoint_stride == 2
    with pytest.raises(ValueError):
        load_local_planner(f"{plugin_module}:DummyPlanner", {"waypoint_stride": 0})


@pytest.mark.parametrize(
    "spec, match",
    [
        ("no_colon_here", "Invalid local planner"),
        (":Missing", "Invalid local planner"),
        ("does_not_exist_xyz:Planner", "Cannot import"),
        ("{mod}:Missing", "has no class"),
        ("{mod}:NotAPlanner", "must be a subclass"),
        ("{mod}:NeedsMissingDep", "missing a dependency"),
    ],
)
def test_bad_specs_fail_with_a_pointer_to_the_docs(plugin_module, spec, match):
    with pytest.raises(PluginError, match=match) as err:
        load_local_planner(spec.format(mod=plugin_module))
    assert "docs/local_planner.md" in str(err.value)


def test_null_planner_builtin():
    planner = load_local_planner("null")
    assert isinstance(planner, NullPlanner)
    assert planner.waypoint_stride == 1
    assert planner.wait_until_ready(timeout=0.1)
    planner.send_waypoint((3.0, -1.0), "odom", yaw=0.5)
    assert planner.last_waypoint == (3.0, -1.0, "odom", 0.5)
    assert planner.status() is PlannerStatus.SUCCEEDED
    planner.cancel()
    assert planner.status() is PlannerStatus.IDLE


def test_builtins_point_to_existing_classes():
    for spec in BUILTIN_PLANNERS.values():
        module, _, cls = spec.partition(":")
        path = Path(__file__).resolve().parents[1] / "ros" / "sem_navae_ros" / "src"
        source = (path / (module.replace(".", "/") + ".py")).read_text()
        assert f"class {cls}(LocalPlanner)" in source
