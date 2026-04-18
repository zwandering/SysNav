cmake_minimum_required(VERSION 3.5)
project(serial)

# Default to C99
if(NOT CMAKE_C_STANDARD)
  set(CMAKE_C_STANDARD 99)
endif()

# Default to C++14
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 14)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)

ament_export_include_directories(include)
ament_export_libraries(${PROJECT_NAME})

add_library(${PROJECT_NAME} SHARED
    src/serial.cc
    include/serial/serial.h
    include/serial/v8stdint.h
)

target_sources(${PROJECT_NAME} PRIVATE
    src/impl/unix.cc
    src/impl/list_ports/list_ports_linux.cc
)
target_link_libraries(${PROJECT_NAME} rt pthread)

target_include_directories(${PROJECT_NAME} PRIVATE include)

install(TARGETS ${PROJECT_NAME}
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
)

install(FILES include/serial/serial.h include/serial/v8stdint.h
    DESTINATION include/serial
)

ament_package()
