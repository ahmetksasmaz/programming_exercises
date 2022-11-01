project(CUDA_EXERCISES LANGUAGES CXX CUDA)

find_package(CUDA REQUIRED)

set(CMAKE_CUDA_ARCHITECTURES 75)

FILE(GLOB subdirs src/*)
FOREACH(subdir ${subdirs})
IF(IS_DIRECTORY ${subdir})
    get_filename_component(dirname ${subdir} NAME_WE)
    file(GLOB ${dirname}_cxx_files ${subdir}/*.cpp)
    file(GLOB ${dirname}_cuda_files ${subdir}/*.cu)
    add_executable(${dirname} ${${dirname}_cxx_files} ${${dirname}_cuda_files})
ENDIF()
ENDFOREACH()