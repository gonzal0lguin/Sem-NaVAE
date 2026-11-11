# Installed by catkin_python_setup(); do not run directly.
from catkin_pkg.python_setup import generate_distutils_setup
from setuptools import setup

setup(
    **generate_distutils_setup(
        packages=["sem_navae_ros", "sem_navae_ros.local_planners"],
        package_dir={"": "src"},
    )
)
