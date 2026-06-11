from setuptools import setup

package_name = 'fusion'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', ['launch/fusion.launch.py']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Sebastian',
    maintainer_email='todo@todo.com',
    description='Drone-immutable + AMR-additive OccupancyGrid fusion node',
    license='MIT',
    entry_points={
        'console_scripts': [
            'map_fusion_node = fusion.fusion_node:main',
        ],
    },
)