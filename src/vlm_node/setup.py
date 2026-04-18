from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'vlm_node'

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
    maintainer='haokun_ros',
    maintainer_email='haokunz@andrew.cmu.edu',
    description='Vision-Language Model reasoning node for object navigation',
    license='PolyForm-Noncommercial-1.0.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'vlm_reasoning_node = vlm_node.vlm_reasoning_node:main',
            'keyboard_input = vlm_node.keyboard_input:main',
        ],
    },
)
