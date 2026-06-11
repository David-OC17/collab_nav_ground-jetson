from setuptools import find_packages, setup

package_name = 'security_integration_test'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
         ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', ['launch/security_test.launch.py']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Fredi Romo',
    maintainer_email='fredi@radix.com.mx',
    description='Integration tests for ros2_security middleware in collab_nav_ground-jetson.',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'stub_scan_pub = security_integration_test.stub_nodes:main_scan',
            'stub_optitrack_pub = security_integration_test.stub_nodes:main_optitrack',
        ],
    },
)
