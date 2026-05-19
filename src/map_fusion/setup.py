import os
from glob import glob
from setuptools import setup

package_name = 'map_fusion'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'),
            glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'config'),
            glob('config/*.yaml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Sebastian',
    maintainer_email='sebastian@example.com',
    description='Drone + SLAM Toolbox OccupancyGrid fusion for Nav2 (ROS 2 Humble).',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'map_fusion_node = map_fusion.map_fusion_node:main',
            'mock_arena = map_fusion.mock_publishers:main',
        ],
    },
)
