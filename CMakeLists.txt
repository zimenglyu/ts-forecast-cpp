cmake_minimum_required(VERSION 3.10)
project(ts_forecast VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Compiler flags for optimization and Raspberry Pi compatibility
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -pedantic")
    set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")
    set(CMAKE_CXX_FLAGS_DEBUG "-g -O0")
endif()

# Option for Raspberry Pi specific optimizations
option(BUILD_FOR_RPI "Enable Raspberry Pi specific optimizations" OFF)
if(BUILD_FOR_RPI)
    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=armv8-a")
    endif()
endif()

# Include directories
include_directories(${CMAKE_SOURCE_DIR}/include)

# Library source files
set(LIB_SOURCES
    src/arima.cpp
    src/exponential_smoothing.cpp
    src/prophet.cpp
    src/gradient_boosting.cpp
    src/dlinear.cpp
    src/utils.cpp
)

# Create static library
add_library(ts_forecast STATIC ${LIB_SOURCES})

# Create shared library
add_library(ts_forecast_shared SHARED ${LIB_SOURCES})
set_target_properties(ts_forecast_shared PROPERTIES OUTPUT_NAME ts_forecast)

# Example executables
add_executable(example_arima examples/example_arima.cpp)
target_link_libraries(example_arima ts_forecast)

add_executable(example_exponential_smoothing examples/example_exponential_smoothing.cpp)
target_link_libraries(example_exponential_smoothing ts_forecast)

add_executable(example_prophet examples/example_prophet.cpp)
target_link_libraries(example_prophet ts_forecast)

add_executable(example_gradient_boosting examples/example_gradient_boosting.cpp)
target_link_libraries(example_gradient_boosting ts_forecast)

add_executable(demo examples/demo.cpp)
target_link_libraries(demo ts_forecast)

add_executable(normalize_etth1 examples/normalize_etth1.cpp)
target_link_libraries(normalize_etth1 ts_forecast)

# Tests
enable_testing()
add_executable(run_tests tests/test_all.cpp)
target_link_libraries(run_tests ts_forecast)
add_test(NAME AllTests COMMAND run_tests)

# ETTh1 Dataset Test
add_executable(test_etth1 tests/test_etth1.cpp)
target_link_libraries(test_etth1 ts_forecast)
add_test(NAME ETTh1Tests COMMAND test_etth1)

# Installation
install(TARGETS ts_forecast ts_forecast_shared
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)
install(DIRECTORY include/ DESTINATION include)
