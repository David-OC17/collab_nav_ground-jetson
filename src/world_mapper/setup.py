from setuptools import setup
import os
from glob import glob

package_name = "world_mapper"

setup(
    name=package_name,
    version="0.1.0",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages",
         ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        (os.path.join("share", package_name, "launch"), glob("launch/*.launch.py")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Sebastian",
    maintainer_email="sebastianopulido04@gmail.com",
    description="Custom occupancy grid mapper in the world frame, using slam_toolbox as a pose source via TF.",
    license="MIT",
    entry_points={
        "console_scripts": [
            "occupancy_mapper = world_mapper.occupancy_mapper:main",
        ],
    },
)