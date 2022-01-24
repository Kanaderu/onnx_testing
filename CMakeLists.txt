cmake_minimum_required(VERSION 3.18.0)

set (PROJECT_NAME "onnx-sample" CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

project(${PROJECT_NAME})

set(default_build_type "Release")
set(ONNXRUNTIME_ROOTDIR $ENV{ONNXRUNTIME_ROOTDIR})
set(Boost_NO_WARN_NEW_VERSIONS 1) # ignore warnings from boost
message(STATUS "ONNXRUNTIME_ROOTDIR: ${ONNXRUNTIME_ROOTDIR}")

#set(Boost_DEBUG ON)
message(STATUS "MSVC_TOOLSET_VERSION is ${MSVC_TOOLSET_VERSION}")

#set(Boost_COMPILER "-vc143") # temp fix for boost on windows (found via debug)

#set (Boost_NO_SYSTEM_PATHS ON)
#set (Boost_USE_MULTITHREADED ON)
set (Boost_USE_STATIC_LIBS ON)
set (Boost_USE_DEBUG_LIBS OFF)
set (Boost_USE_RELEASE_LIBS ON)
#set (Boost_USE_STATIC_RUNTIME OFF)
#set (BOOST_ALL_DYN_LINK OFF)

find_package(OpenCV REQUIRED)
find_package(Boost REQUIRED COMPONENTS system filesystem log log_setup program_options)
message(STATUS "Boost version: ${Boost_VERSION}")

include_directories(${Boost_INCLUDE_DIR})

add_executable(${PROJECT_NAME} main.cpp)
target_link_libraries (${PROJECT_NAME} ${Boost_LIBRARIES})

if (APPLE)
  target_include_directories(${PROJECT_NAME} PRIVATE "${ONNXRUNTIME_ROOTDIR}/include/onnxruntime/core/session")
else()
  target_include_directories(${PROJECT_NAME} PRIVATE "${ONNXRUNTIME_ROOTDIR}/include")
endif()

target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})

if (WIN32)
  target_link_libraries(${PROJECT_NAME} "${ONNXRUNTIME_ROOTDIR}/lib/onnxruntime.lib")
endif()

if (APPLE)
  target_link_libraries(${PROJECT_NAME} "${ONNXRUNTIME_ROOTDIR}/lib/libonnxruntime.dylib")
endif()

if (UNIX AND NOT APPLE)
  target_link_libraries(${PROJECT_NAME} "${ONNXRUNTIME_ROOTDIR}/lib/libonnxruntime.so")
endif()
