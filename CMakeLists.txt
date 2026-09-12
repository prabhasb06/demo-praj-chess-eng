cmake_minimum_required(VERSION 3.12)
project(AntigravityChess CXX)

option(BUILD_EXPERIMENTAL_CPP "Build the non-playing C++ reference sketch" OFF)
if(NOT BUILD_EXPERIMENTAL_CPP)
    message(STATUS "Production engine is Python: run python main.py. C++ sketch excluded.")
    return()
endif()
message(WARNING "Experimental C++ sketch has no legal move generator; never use it for play.")

# Require C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Check for environment prefix path to LibTorch
if(DEFINED ENV{LIBTORCH_PATH})
    list(APPEND CMAKE_PREFIX_PATH $ENV{LIBTORCH_PATH})
endif()

# Find LibTorch (will look for TorchConfig.cmake in prefix path)
find_package(Torch QUIET)

if(Torch_FOUND)
    message(STATUS "LibTorch found at ${TORCH_INSTALL_PREFIX}. Building with deep neural network support.")
    add_executable(antigravity_chess uci_engine.cpp)
    target_compile_definitions(antigravity_chess PRIVATE USE_LIBTORCH)
    target_link_libraries(antigravity_chess "${TORCH_LIBRARIES}")
    if(MSVC)
        target_compile_options(antigravity_chess PRIVATE /wd4624 /wd4267) # Suppress warnings
    endif()
else()
    message(STATUS "LibTorch not found. Compiling engine in standard/fallback mode.")
    add_executable(antigravity_chess uci_engine.cpp)
endif()
