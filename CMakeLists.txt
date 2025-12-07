cmake_minimum_required(VERSION 3.14)
project(cqt VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Source files - Core
set(CORE_SOURCES
    src/core/fft.cpp
    src/core/resample.cpp
)

set(CORE_HEADERS
    src/core/fft.hpp
    src/core/resample.hpp
)

# Source files - CQT variants
set(CQT_SOURCES
    src/cqt/hybrid_cqt.cpp
    src/cqt/standard_cqt.cpp
    src/cqt/pseudo_cqt.cpp
    src/cqt/vqt.cpp
)

set(CQT_HEADERS
    src/cqt/cqt_base.hpp
    src/cqt/hybrid_cqt.hpp
    src/cqt/standard_cqt.hpp
    src/cqt/pseudo_cqt.hpp
    src/cqt/vqt.hpp
)

# Source files - Utilities
set(UTILS_SOURCES
    src/utils/convert.cpp
)

set(UTILS_HEADERS
    src/utils/convert.hpp
)

# Other headers
set(OTHER_HEADERS
    src/progress.hpp
)

# All sources combined
set(ALL_SOURCES
    ${CORE_SOURCES}
    ${CQT_SOURCES}
    ${UTILS_SOURCES}
)

set(ALL_HEADERS
    ${CORE_HEADERS}
    ${CQT_HEADERS}
    ${UTILS_HEADERS}
    ${OTHER_HEADERS}
)

# Option for building with Emscripten
option(BUILD_WASM "Build WebAssembly version" OFF)

if(BUILD_WASM)
    # WebAssembly build
    add_executable(cqt_wasm ${ALL_SOURCES} src/bindings.cpp)

    set_target_properties(cqt_wasm PROPERTIES
        OUTPUT_NAME "cqt"
        SUFFIX ".js"
    )

    target_include_directories(cqt_wasm PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
    )

    target_compile_options(cqt_wasm PRIVATE
        -O3
        -ffast-math
    )

    target_link_options(cqt_wasm PRIVATE
        -sWASM=1
        -sMODULARIZE=1
        -sEXPORT_NAME="createCQTModule"
        -sALLOW_MEMORY_GROWTH=1
        -sEXPORTED_RUNTIME_METHODS=['ccall','cwrap']
        -sENVIRONMENT=web,worker
        -sEXPORT_ES6=1
        --bind
        -O3
    )

    # Copy output to dist directory
    add_custom_command(TARGET cqt_wasm POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy
            $<TARGET_FILE:cqt_wasm>
            ${CMAKE_SOURCE_DIR}/dist/cqt.js
        COMMAND ${CMAKE_COMMAND} -E copy
            ${CMAKE_BINARY_DIR}/cqt.wasm
            ${CMAKE_SOURCE_DIR}/dist/cqt.wasm
    )
else()
    # Native library build
    add_library(cqt STATIC ${ALL_SOURCES})

    target_include_directories(cqt PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>
        $<INSTALL_INTERFACE:include>
    )

    # Test executable
    add_executable(test_cqt test/test_cqt.cpp)
    target_link_libraries(test_cqt PRIVATE cqt)

    # Enable testing
    enable_testing()
    add_test(NAME TestCQT COMMAND test_cqt)
endif()
