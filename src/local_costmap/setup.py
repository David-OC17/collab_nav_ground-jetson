from setuptools import setup

package_name = 'local_costmap'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/config', ['config/local_costmap_params.yaml']),
        ('share/' + package_name + '/launch', ['launch/local_costmap_launch.py']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='rover',
    description='Local costmap node with obstacle marking, raytracing and inflation',
    entry_points={
        'console_scripts': [
            'local_costmap_node = local_costmap.local_costmap_node:main',
            'fake_map_publisher = local_costmap.fake_map_publisher:main',
        ],
    },
)