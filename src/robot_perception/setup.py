from glob import glob

from setuptools import find_packages, setup

package_name = 'robot_perception'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch',
            glob('launch/*.launch.py')),
        ('share/' + package_name + '/config',
            glob('config/*.yaml')),
        ('share/' + package_name + '/models',
            glob('models/*.pt')),
    ],
    scripts=['scripts/yolo_target_detector'],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='EveningStudy',
    maintainer_email='EveningStudy@users.noreply.github.com',
    description='RGB-D perception nodes for the autonomy lab robot.',
    license='AGPL-3.0-only',
    entry_points={
        'console_scripts': [
            (
                'rgbd_inspector = '
                'robot_perception.rgbd_inspector_node:main'
            ),
            (
                'red_target_detector = '
                'robot_perception.red_target_detector_node:main'
            ),
            (
                'yolo_dataset_collector = '
                'robot_perception.yolo_dataset_collector_node:main'
            ),
            (
                'yolo_dataset_report = '
                'robot_perception.yolo_dataset_report:main'
            ),
        ],
    },
)
