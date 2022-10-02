cmake_minimum_required(VERSION 3.7)
project(InterfaceGrid VERSION 1.0 LANGUAGES CXX)
set(CMAKE_VERBOSE_MAKEFILE on)

include(GNUInstallDirs)
set(CMAKE_CXX_STANDARD 14)
set(LIB /cs/labs/dina/punims/gamb)
set(MYLIB /cs/labs/dina/punims/DockingLib)
set(CMAKE_PREFIX_PATH /cs/labs/dina/punims/)
set(HighFive_DIR /cs/labs/dina/punims/HighFive/CMake)
set(HighFive_path /cs/labs/dina/punims/HighFive/include)
set(HIGHFIVE_USE_INSTALL_DEPS=YES)
find_package(HighFive 2.0 REQUIRED)

set(CMAKE_FIND_LIBRARY_SUFFIXES ".a")
set(BUILD_SHARED_LIBS OFF)
set(ENABLE_STATIC ON)
set(CMAKE_EXE_LINKER_FLAGS "-static")
set(Boost_USE_STATIC_LIBS ON)
set(HDF5_USE_STATIC_LIBRARIES ON)

FIND_PACKAGE(Boost COMPONENTS program_options system filesystem REQUIRED)
INCLUDE_DIRECTORIES(${LIB} ${MYLIB} ${Boost_INCLUDE_DIRS}b ${HighFive_path} /usr/include/hdf5/serial )
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -O2")

file(GLOB SOURCE_FILES_GRID
        GraphInterface.cpp InterfaceGrid.cc MolecularInterface.cc interface.cc
        )
link_directories(${LIB} ${MYLIB} /usr/lib/x86_64-linux-gnu/ /usr/lib/x86_64-linux-gnu/hdf5/serial)
add_executable(interface2grid ${SOURCE_FILES_GRID})
target_link_libraries(interface2grid hdf5 z dockingLib gamb++ boost_program_options boost_system boost_filesystem m)
