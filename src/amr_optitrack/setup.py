from setuptools import find_packages, setup

package_name = 'amr_optitrack'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='noe',
    maintainer_email='noe.benjamin2010@hotmail.com',
    description='Your package description here',
    license='Apache-2.0',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'optitrack_pose_node = amr_optitrack.optitrack_pose_node:main',
        ],
    },
)
