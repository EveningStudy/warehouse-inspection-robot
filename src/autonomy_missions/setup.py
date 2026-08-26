from setuptools import find_packages, setup

package_name = 'autonomy_missions'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/config',
            [
                'config/warehouse_patrol.yaml',
                'config/yolo_collection_route.yaml',
                'config/inspection_patrol.yaml',
            ]),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='EveningStudy',
    maintainer_email='EveningStudy@users.noreply.github.com',
    description='Mission control nodes for the autonomy lab robot.',
    license='AGPL-3.0-only',
    entry_points={
        'console_scripts': [
            (
                'single_goal = '
                'autonomy_missions.single_goal_node:main'
            ),
            (
                'patrol = '
                'autonomy_missions.patrol_node:main'
            ),
            (
                'warehouse_patrol = '
                'autonomy_missions.warehouse_patrol_node:main'
            ),
            (
                'approach_target = '
                'autonomy_missions.approach_target_node:main'
            ),
            (
                'warehouse_mapper = '
                'autonomy_missions.warehouse_mapper_node:main'
            ),
            (
                'inspection_patrol = '
                'autonomy_missions.inspection_patrol_node:main'
            ),
        ],
    },
)
