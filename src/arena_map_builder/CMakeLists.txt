cmake_minimum_required(VERSION 3.8...3.22)
project(arena_map_builder)

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(arena_map_builder_msgs REQUIRED)

# Python package (nodes live in setup.py / setup.cfg)
ament_python_install_package(${PROJECT_NAME})

install(PROGRAMS
  scripts/build_arena_map_server
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY launch
  DESTINATION share/${PROJECT_NAME}/
  OPTIONAL
)
install(DIRECTORY config
  DESTINATION share/${PROJECT_NAME}/
  OPTIONAL
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()