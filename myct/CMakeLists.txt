cmake_minimum_required(VERSION 3.8) # 确保使用支持 C++17 的 CMake 版本
project(CTRANSLATE2_DEMO)

# 使用 C++17 标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 包含头文件目录
include_directories(include)  # 包含根目录的 include
include_directories(include/sentencepiece)  # 特别包含 sentencepiece 的头文件路径

# 设置输出可执行文件名
set(OUT_BIN_NAME "main")

# 设置库路径
link_directories(lib)  # 指向项目内的 lib 文件夹

# 指定源文件
add_executable("${OUT_BIN_NAME}" main.cpp)

# 链接库
set(ct_lib ctranslate2)
set(sp_lib sentencepiece)

# 将库链接到目标
target_link_libraries(main ${ct_lib} ${sp_lib})
