#!/usr/bin/env python3

from setuptools import setup, find_packages
import os
from glob import glob

package_name = 'car_pkg'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(),
    data_files=[
        ('share/' + package_name, ['package.xml']),
        
        # Resources (images, templates, etc.)
        (os.path.join('share', package_name, 'resources'), 
         glob('resources/*.png') + glob('resources/*.jpg') + glob('resources/*.jpeg')),
        
        # Launch files
        (os.path.join('share', package_name, 'launch'), 
         glob('launch/*.launch.py') + glob('launch/*.py')),
        # Augmentation script
        (os.path.join('share', package_name), ['create_augmented.py']),
        (os.path.join('share', package_name, 'world'), glob('world/*')),
    ],
    install_requires=['setuptools', 'opencv-python', 'numpy'],
    zip_safe=True,
    maintainer='Leonardo Chen',
    maintainer_email='leochenjin@gmail.com',
    description='Car package with sign detection',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'webots_bridge = car_pkg.webots_bridge:main',
            'road_follower = car_pkg.road_follower:main',
            'sign_detector = car_pkg.sign_detector:main',
            'car_controller = car_pkg.car_controller:main',
        ],
    },
)
