import os
from glob import glob

from setuptools import find_packages, setup

package_name = 'arena_map_builder'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/arena_map_builder']),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/data/models',
            glob(os.path.join('data', 'models', '*.onnx'))),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='david',
    maintainer_email='daveoc01@icloud.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'build_arena_map_server = arena_map_builder.build_arena_map_server:main',
            'example_client = arena_map_builder.example_client:main',
        ],
    },
)