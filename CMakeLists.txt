cmake_minimum_required(VERSION 3.14)
project(n_pendulum)

set(CMAKE_CXX_STANDARD 20)

include(FetchContent)
FetchContent_Declare(
    eigen
    GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
    GIT_TAG        3.4.0
    GIT_SHALLOW    TRUE
)
FetchContent_MakeAvailable(eigen)

add_executable(n_pendulum n_pendulum.cpp)
target_link_libraries(n_pendulum PRIVATE Eigen3::Eigen)