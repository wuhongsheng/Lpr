cmake_minimum_required(VERSION 3.22)
project(Lpr)

set(CMAKE_CXX_STANDARD 14)

# OpenCV
find_package(OpenCV REQUIRED)
if (OpenCV_FOUND)
    # If the package has been found, several variables will
    # be set, you can find the full list with descriptions
    # in the OpenCVConfig.cmake file.
    # Print some message showing some of them
    message(STATUS "OpenCV library status:")
    message(STATUS "    version: ${OpenCV_VERSION}")
    message(STATUS "    include path: ${OpenCV_INCLUDE_DIRS}" \n)
else ()
    message(FATAL_ERROR "Could not locate OpenCV" \n)
endif()

# OpenVINO
find_package(OpenVINO REQUIRED)
if(CMAKE_COMPILER_IS_GNUCXX)
    ov_add_compiler_flags(-Wall)
endif()

if(APPLE)
    set(CMAKE_MACOSX_RPATH ON)
endif()


file(GLOB LPR_SOURCE_FILES src/LPR/*.cpp src/LPR/*.h src/LPR/*.hpp)

add_executable(Lpr src/main.cpp ${LPR_SOURCE_FILES})


target_link_libraries (
        Lpr
        ${OpenCV_LIBS}
        openvino::runtime
)