cmake_minimum_required(VERSION 3.8)

project(diff_drive_description)

find_package(ament_cmake REQUIRED)

install(
  DIRECTORY
    config
    launch
    maps
    meshes
    models
    rviz
    urdf
    worlds
  DESTINATION share/${PROJECT_NAME}
)

install(
  PROGRAMS
    scripts/odom_adapter.py
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()
