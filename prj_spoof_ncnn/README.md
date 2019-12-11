- 在CMakeLists.txt中添加include_directories以添加对应的头文件路径

```C
# cmake_minimum_required(VERSION 2)
project(demo_ncnn)

# set(CMAKE_CXX_STANDARD 14)

add_executable(demo_ncnn main.cpp)
include_directories(
        /usr/local/include/opencv4
        /home/cmf/dev/ncnn/src
        /home/cmf/dev/ncnn/build/src
)
#link_directories(
#        /home/cmf/dev/ncnn/build/src
#)
```

- 在VSCode中的c_cpp_properties.json中添加对应的头文件路径以让VSCode识别到对应的头文件路径

```json
{
    "configurations": [
        {
            "name": "Linux",
            "includePath": [
                "/usr/local/include/opencv4/**",
                "/home/cmf/dev/ncnn/src/**",
                "/home/cmf/dev/ncnn/build/src/**",
                "${workspaceFolder}/**"
            ],
            "defines": [],
            "compilerPath": "/usr/bin/gcc",
            "cStandard": "c11",
            "cppStandard": "c++17",
            "intelliSenseMode": "clang-x64"
        }
    ],
    "version": 4
}
```

- 运行

```sh
mkdir build
cd build
cmake
make
./untitled
```