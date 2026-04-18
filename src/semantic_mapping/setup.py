from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'semantic_mapping'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name), glob('launch/*.launch')),
        (os.path.join('share', package_name), glob('config/*.yaml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Zihan Liu',
    maintainer_email='zihanli4@andrew.cmu.edu',
    description='Semantic mapping node for TARE planner',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
             'semantic_mapping_node = semantic_mapping.semantic_mapping_node:main',
             'detection_node = semantic_mapping.detection_node:main'
        ],
    },
)