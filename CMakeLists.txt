cmake_minimum_required(VERSION 3.15)
project(simpli_search VERSION 1.0.0 LANGUAGES CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Build type
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Compiler flags
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -O0 -Wall -Wextra")
else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")
endif()

# Find required packages
find_package(Eigen3 REQUIRED)
find_package(CURL REQUIRED)
find_package(cpprestsdk REQUIRED)
find_package(PostgreSQL REQUIRED)
find_package(nlohmann_json 3.10.5 REQUIRED)

# ONNX Runtime setup
set(ONNXRUNTIME_ROOTDIR "/usr/local" CACHE PATH "Root directory of ONNX Runtime installation")
set(ONNXRUNTIME_INCLUDE_DIRS "${ONNXRUNTIME_ROOTDIR}/include/onnxruntime")
set(ONNXRUNTIME_LIB_DIRS "${ONNXRUNTIME_ROOTDIR}/lib")

# Verify ONNX Runtime files exist
if(NOT EXISTS "${ONNXRUNTIME_INCLUDE_DIRS}/onnxruntime_cxx_api.h")
    message(FATAL_ERROR "ONNX Runtime headers not found in ${ONNXRUNTIME_INCLUDE_DIRS}")
endif()

# Source files
set(SOURCES
    ./simpli_search/simpli_search.cpp
)

# Create executable
add_executable(simpli_search_server ${SOURCES})

# Include directories
target_include_directories(simpli_search_server
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${ONNXRUNTIME_INCLUDE_DIRS}
        ${EIGEN3_INCLUDE_DIRS}
        ${PostgreSQL_INCLUDE_DIRS}
)

# Link directories
target_link_directories(simpli_search_server
    PRIVATE
        ${ONNXRUNTIME_LIB_DIRS}
)

# Link libraries
target_link_libraries(simpli_search_server
    PRIVATE
        onnxruntime
        Eigen3::Eigen
        CURL::libcurl
        cpprestsdk::cpprest
        PostgreSQL::PostgreSQL
        nlohmann_json::nlohmann_json
        pthread
        crypto
)

# Installation
install(TARGETS simpli_search_server
    RUNTIME DESTINATION bin
)

# Print configuration summary
message(STATUS "")
message(STATUS "Configuration summary:")
message(STATUS "  C++ Compiler: ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")
message(STATUS "  Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "  ONNX Runtime include path: ${ONNXRUNTIME_INCLUDE_DIRS}")
message(STATUS "  ONNX Runtime library path: ${ONNXRUNTIME_LIB_DIRS}")
message(STATUS "  PostgreSQL version: ${PostgreSQL_VERSION_STRING}")
message(STATUS "  Eigen3 version: ${EIGEN3_VERSION_STRING}")
message(STATUS "")

# Add testing support if needed
# enable_testing()
# add_subdirectory(tests)