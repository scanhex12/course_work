cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(untitled1)



set(CMAKE_PREFIX_PATH "libtorch/share/cmake/Torch")

find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

add_executable(untitled1 main.cpp
        fastknn.cpp fastknn.h
        queue_recommendations.cpp queue_recommendations.h
        metrics.h
        embedding_parser.cpp embedding_parser.h
        csv_parser.cpp csv_parser.h
        progress_bar.h
        inference.h
        logger.h)
target_link_libraries(untitled1 "${TORCH_LIBRARIES}")
set_property(TARGET untitled1 PROPERTY CXX_STANDARD 14)
add_subdirectory(tests)
# The following code block is suggested to be used on Windows.
# According to https://github.com/pytorch/pytorch/issues/25457,
# the DLLs need to be copied to avoid memory errors.
if (MSVC)
    file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
    add_custom_command(TARGET untitled1
            POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
            ${TORCH_DLLS}
            $<TARGET_FILE_DIR:untitled1>)
endif (MSVC)

