from setuptools import setup
import os
from glob import glob

package_name = 'yahboomcar_laser'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share',package_name,'launch'),glob(os.path.join('launch','*launch.py')))
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='yahboom',
    maintainer_email='yahboom@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
        'laser_avoidance = yahboomcar_laser.laser_avoidance:main',
        'laser_tracker = yahboomcar_laser.laser_tracker:main',
        'laser_warning = yahboomcar_laser.laser_warning:main',
        ],
    },
)
