cmake_minimum_required(VERSION 3.10)
project(arena_marker_localizer)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)

ament_python_install_package(${PROJECT_NAME})

install(PROGRAMS
  scripts/marker_localizer_service
  scripts/check_sync
  scripts/calibrate_extrinsics
  scripts/calibrate_bias
  scripts/calibrate_bias_v2
  scripts/calibrate_bias_v3
  scripts/calibrate_bias_sweep
  scripts/transform_telemetry
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