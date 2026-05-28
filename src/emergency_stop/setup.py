from setuptools import setup
import os
from glob import glob

setup(
    name='emergency_stop',
    version='0.0.1',
    packages=['emergency_stop'],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/emergency_stop']),
        ('share/emergency_stop', ['package.xml']),
        # ← This line installs all launch files
        (os.path.join('share', 'emergency_stop', 'launch'),
            glob('launch/*.py')),
    ],
    install_requires=['setuptools'],
    entry_points={
        'console_scripts': [
            'emergency_stop = emergency_stop.emergency_stop:main',
        ],
    },
)