import os
from glob import glob
from setuptools import find_packages, setup

package_name = 'frontier_explorer'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),

        (os.path.join('share', package_name, 'launch'),
            glob('launch/*.py')),
        (os.path.join('share', package_name, 'rviz'),
            glob('rviz/*.rviz')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='polo',
    maintainer_email='jorglezd28@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            # Core nodes
            'frontier_explorer   = frontier_explorer.frontier_explorer:main',
            'explorer_controller = frontier_explorer.explorer_controller:main',
            'aruco_goal_detector = frontier_explorer.aruco_goal_detector:main',
            'aruco_world_bridge  = frontier_explorer.aruco_world_bridge:main',
            'camera_fov_tracker  = frontier_explorer.camera_fov_tracker:main',
             'aruco_visual_servo  = frontier_explorer.aruco_visual_servo:main',
            # Simulation helpers — now in simulation_helpers subpackage
            'fake_aruco_detector = frontier_explorer.simulation_helpers.fake_aruco_detector:main',
            'fake_map_publisher  = frontier_explorer.simulation_helpers.fake_map_publisher:main',
            'fake_pose_publisher = frontier_explorer.simulation_helpers.fake_pose_publisher:main',
            'odom_to_pose        = frontier_explorer.simulation_helpers.odom_to_pose:main',
            'pose_to_tf          = frontier_explorer.simulation_helpers.pose_to_tf:main',
        ],
    },
)
