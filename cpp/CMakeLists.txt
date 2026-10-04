#
# C++ Agentcore：按业务目录自动纳入源文件，新增一个实现文件不需要维护注册清单。
# 运行时只依赖系统的 libcurl 和 JsonCpp；公开接口保持在 model、agent、openai 三个目录。
#
cmake_minimum_required(VERSION 3.20)

project(agentcore_cpp VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

find_package(PkgConfig REQUIRED)
pkg_check_modules(JSONCPP REQUIRED IMPORTED_TARGET jsoncpp)
pkg_check_modules(CURL REQUIRED IMPORTED_TARGET libcurl)

# 目录就是模块边界；CONFIGURE_DEPENDS 让放入目录的新文件自动参加构建。
file(GLOB_RECURSE AGENTCORE_SOURCES CONFIGURE_DEPENDS
     "${CMAKE_CURRENT_SOURCE_DIR}/agent/*.cpp"
     "${CMAKE_CURRENT_SOURCE_DIR}/model/*.cpp"
     "${CMAKE_CURRENT_SOURCE_DIR}/openai/*.cpp")

add_library(agentcore_cpp STATIC ${AGENTCORE_SOURCES})
target_include_directories(agentcore_cpp PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}")
target_link_libraries(agentcore_cpp PUBLIC PkgConfig::JSONCPP PkgConfig::CURL)
target_compile_options(agentcore_cpp PRIVATE -Wall -Wextra -Wpedantic)

include(CTest)
if(BUILD_TESTING)
    add_executable(agentcore_cpp_tests tests/core_test.cpp)
    target_link_libraries(agentcore_cpp_tests PRIVATE agentcore_cpp)
    target_compile_options(agentcore_cpp_tests PRIVATE -Wall -Wextra -Wpedantic)
    add_test(NAME agentcore_cpp_tests COMMAND agentcore_cpp_tests)
endif()
