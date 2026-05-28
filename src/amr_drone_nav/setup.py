from setuptools import setup
from glob import glob
import os

package_name = 'amr_drone_nav'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    data_files=[
        # ament index marker — required for ROS 2 to discover the package
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        # package.xml
        (os.path.join('share', package_name), ['package.xml']),
        # launch files
        (os.path.join('share', package_name, 'launch'),
            glob('launch/*.launch.py')),
        # config files
        (os.path.join('share', package_name, 'config'),
            glob('config/*.yaml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Sebastian',
    maintainer_email='sebastianopulido04@gmail.com',
    description='AMR + drone map fusion bring-up for Nav2.',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'alignment_node = amr_drone_nav.alignment_node:main',
            'stub_drone_map_publisher = '
                'amr_drone_nav.stub_drone_map_publisher:main',
            'stub_slam_map_publisher = amr_drone_nav.stub_slam_map_publisher:main',
        ],
    },
)
