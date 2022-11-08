project(OPENCV_EXERCISES LANGUAGES CXX CUDA)

find_package(CUDA REQUIRED)
find_package(OpenCV REQUIRED)

set(CMAKE_CUDA_ARCHITECTURES 75)

file(GLOB SUBDIRS src/*)
foreach(SUBDIR ${SUBDIRS})
if(IS_DIRECTORY ${SUBDIR})
    get_filename_component(DIRNAME ${SUBDIR} NAME_WE)
    file(GLOB ${DIRNAME}_CXX_FILES ${SUBDIR}/*.cpp)
    file(GLOB ${DIRNAME}_CUDA_FILES ${SUBDIR}/*.cu)
    add_executable(${DIRNAME} ${${DIRNAME}_CXX_FILES} ${${DIRNAME}_CUDA_FILES})
    target_include_directories(${DIRNAME} ${OpenCV_INCLUDE_DIRS})
    target_link_libraries(${DIRNAME} ${OpenCV_LIBS})
endif()
endforeach()