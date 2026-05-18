cmake_minimum_required(VERSION 3.8)
project(arena_map_builder_msgs)

find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(builtin_interfaces REQUIRED)
find_package(action_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "action/BuildArenaMap.action"
  DEPENDENCIES nav_msgs std_msgs geometry_msgs builtin_interfaces action_msgs
)

ament_package()