"""Local planner plugins. ``load_local_planner("pkg.module:Class", params)`` builds one."""

from __future__ import annotations

import importlib
from typing import Any, Dict, Optional

from sem_navae_ros.local_planners.base import LocalPlanner, PlannerStatus

BUILTIN_PLANNERS = {
    "move_base": "sem_navae_ros.local_planners.move_base:MoveBasePlanner",
    "null": "sem_navae_ros.local_planners.null:NullPlanner",
}

_HELP = "See docs/local_planner.md for how to write a local planner plugin."


class PluginError(RuntimeError):
    pass


def load_local_planner(spec: str, params: Optional[Dict[str, Any]] = None) -> LocalPlanner:
    """Instantiate a planner from a built-in name or a ``"package.module:ClassName"`` path."""
    path = BUILTIN_PLANNERS.get(spec, spec)
    module_name, sep, class_name = path.partition(":")
    if not sep or not module_name or not class_name:
        raise PluginError(
            f"Invalid local planner '{spec}': use one of {sorted(BUILTIN_PLANNERS)} "
            f"or 'package.module:ClassName'. {_HELP}"
        )
    try:
        module = importlib.import_module(module_name)
    except ImportError as e:
        raise PluginError(
            f"Cannot import local planner module '{module_name}': {e}. {_HELP}"
        ) from e
    cls = getattr(module, class_name, None)
    if cls is None:
        raise PluginError(f"Module '{module_name}' has no class '{class_name}'. {_HELP}")
    if not (isinstance(cls, type) and issubclass(cls, LocalPlanner)):
        raise PluginError(
            f"'{path}' must be a subclass of sem_navae_ros.local_planners.LocalPlanner. {_HELP}"
        )
    try:
        return cls(params or {})
    except ImportError as e:  # adapters import their ROS dependencies in __init__
        raise PluginError(f"Local planner '{spec}' is missing a dependency: {e}. {_HELP}") from e


__all__ = ["BUILTIN_PLANNERS", "LocalPlanner", "PlannerStatus", "PluginError", "load_local_planner"]
