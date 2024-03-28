cmake_minimum_required(VERSION 3.5)

set(PROJECT_NAME Yolov8OnnxRuntimeCPPInference)
project(${PROJECT_NAME} VERSION 0.0.1 LANGUAGES CXX)


# -------------- Support C++17 for using filesystem  ------------------#
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS ON)
set(CMAKE_INCLUDE_CURRENT_DIR ON)
message("c_CMAKE_CURRENT_SOURCE_DIR:  " ${CMAKE_CURRENT_SOURCE_DIR} )


# -------------- OpenCV  ------------------#
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})


# -------------- Compile CUDA for FP16 inference if needed  ------------------#
option(USE_CUDA "Enable CUDA support" ON)
if (NOT APPLE AND USE_CUDA)
    find_package(CUDA REQUIRED)
    include_directories(${CUDA_INCLUDE_DIRS})
    add_definitions(-DUSE_CUDA)
else ()
    set(USE_CUDA OFF)
endif ()

# -------------- ONNXRUNTIME  ------------------#
set(ONNXRUNTIME_VERSION 1.17.0)

if (WIN32)
    if (USE_CUDA)
        set(ONNXRUNTIME_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/onnxruntime-win-gpu-${ONNXRUNTIME_VERSION}")
    else ()
        set(ONNXRUNTIME_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/onnxruntime-win-cpu-${ONNXRUNTIME_VERSION}")
    endif ()
elseif (LINUX)
    if (USE_CUDA)
        set(ONNXRUNTIME_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/onnxruntime-linux-gpu-${ONNXRUNTIME_VERSION}")
    else ()
        set(ONNXRUNTIME_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/onnxruntime-linux-cpu-${ONNXRUNTIME_VERSION}")
    endif ()
endif ()

include_directories(${PROJECT_NAME} ${ONNXRUNTIME_ROOT}/include)

set(PROJECT_SOURCES inference.h inference.cpp main.cpp)

add_executable(${PROJECT_NAME} ${PROJECT_SOURCES})

if (WIN32)
    target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS} ${ONNXRUNTIME_ROOT}/lib/onnxruntime.lib)
    if (USE_CUDA)
        target_link_libraries(${PROJECT_NAME} ${CUDA_LIBRARIES})
    endif ()
elseif (LINUX)
    target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS} 
                                          ${ONNXRUNTIME_ROOT}/lib/libonnxruntime.so
                                          ${ONNXRUNTIME_ROOT}/lib/libonnxruntime.so.1.17.0
                                          ${ONNXRUNTIME_ROOT}/lib/libonnxruntime_providers_cuda.so
                                          ${ONNXRUNTIME_ROOT}/lib/libonnxruntime_providers_shared.so)
    if (USE_CUDA)
        target_link_libraries(${PROJECT_NAME} ${CUDA_LIBRARIES})
    endif ()
endif ()

