from setuptools import find_packages, setup

package_name = 'trajectory_planner'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
    ('share/ament_index/resource_index/packages', ['resource/trajectory_planner']),
    ('share/trajectory_planner', ['package.xml']),
    ('share/trajectory_planner/launch', ['launch/trajectory_planner_launch.py']),
    ('share/trajectory_planner/config', ['config/trajectory_planner.rviz']),
],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='jetson',
    maintainer_email='daveoc1704@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'astar_planner = trajectory_planner.astar_planner:main',
            'astar_planner2 = trajectory_planner.astar_planner2:main',
            'spline_follower = trajectory_planner.spline_follower:main',
            'trajectory_follower_sim  = trajectory_planner.trajectory_follower_sim:main',
        ],
    },
)
