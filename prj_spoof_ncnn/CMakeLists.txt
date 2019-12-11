cmake_minimum_required(VERSION 2.8)
project(untitled)

#set(CMAKE_CXX_STANDARD 14)

# 配置ncnn的链接库和头文件
set(NCNN_LIBS /home/cmf/dev/ncnn-20191113/build-host-gcc-linux-without-openmp/install/lib/libncnn.a)
set(NCNN_INCLUDE_DIRS /home/cmf/dev/ncnn-20191113/build-host-gcc-linux-without-openmp/install/include/ncnn)
#add_subdirectory(arcface)
file(GLOB SRC_LIST "jni.cpp" "arcface/*.cpp" )
file(GLOB CLASSIFY_LIST  "classify/*.cpp")
# 配置opencv
find_package(OpenCV REQUIRED)
message(${OpenCV_INCLUDE_DIRS})
message(${OpenCV_LIBS})

#set(${OpenCV_INCLUDE_DIRS} )

#添加openmp
FIND_PACKAGE( OpenMP REQUIRED)
if(OPENMP_FOUND)
    message("OPENMP FOUND")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

# VULKAN
#find_package(Vulkan REQUIRED)
#message(${Vulkan_INCLUDE_DIRS})
#message(${Vulkan_LIBRARIES})

include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${NCNN_INCLUDE_DIRS})
#include_directories(arcface)

# 建立链接依赖
add_executable(main ${SRC_LIST} "main.cpp")
target_link_libraries(main ${OpenCV_LIBS})
target_link_libraries(main ${NCNN_LIBS})

add_executable(compare ${SRC_LIST} "compare.cpp")
target_link_libraries(compare ${OpenCV_LIBS})
target_link_libraries(compare ${NCNN_LIBS})

add_executable(scale_test scale_test.cpp)

add_executable(infer_det_on_video ${SRC_LIST} "infer_det_on_video.cpp")
target_link_libraries(infer_det_on_video ${OpenCV_LIBS})
target_link_libraries(infer_det_on_video ${NCNN_LIBS})


add_executable(video_player video_player.cpp)
target_link_libraries(video_player ${OpenCV_LIBS})
#target_link_libraries(video_player ${Vulkan_LIBRARIES})

add_executable(infer_classify_on_video infer_classify_on_video.cpp)
target_link_libraries(infer_classify_on_video ${OpenCV_LIBS})
target_link_libraries(infer_classify_on_video ${NCNN_LIBS})

add_executable(infer_det_classify_on_video ${SRC_LIST} infer_det_classify_on_video.cpp)
target_link_libraries(infer_det_classify_on_video ${OpenCV_LIBS})
target_link_libraries(infer_det_classify_on_video ${NCNN_LIBS})

add_executable(infer_classifier_on_video ${SRC_LIST} ${CLASSIFY_LIST} infer_classifier_on_video.cpp)
target_link_libraries(infer_classifier_on_video ${OpenCV_LIBS})
target_link_libraries(infer_classifier_on_video ${NCNN_LIBS})


add_executable(infer_classify_on_image infer_classify_on_image.cpp)
target_link_libraries(infer_classify_on_image ${OpenCV_LIBS})
target_link_libraries(infer_classify_on_image ${NCNN_LIBS})

add_executable(infer_det_classifier_on_video ${SRC_LIST} ${CLASSIFY_LIST} infer_det_classifier_on_video.cpp)
target_link_libraries(infer_det_classifier_on_video ${OpenCV_LIBS})
target_link_libraries(infer_det_classifier_on_video ${NCNN_LIBS})

add_executable(infer_classifier_on_image ${CLASSIFY_LIST} infer_classifier_on_image.cpp)
target_link_libraries(infer_classifier_on_image ${OpenCV_LIBS})
target_link_libraries(infer_classifier_on_image ${NCNN_LIBS})

add_executable(infer_det_classifier_on_image ${SRC_LIST} ${CLASSIFY_LIST} infer_det_classifier_on_image.cpp)
target_link_libraries(infer_det_classifier_on_image ${OpenCV_LIBS})
target_link_libraries(infer_det_classifier_on_image ${NCNN_LIBS})

add_executable(infer_det_classify_on_image ${SRC_LIST} ${CLASSIFY_LIST} infer_det_classify_on_image.cpp)
target_link_libraries(infer_det_classify_on_image ${OpenCV_LIBS})
target_link_libraries(infer_det_classify_on_image ${NCNN_LIBS})