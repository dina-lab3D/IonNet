cmake_minimum_required(VERSION 3.7)
project(InterfaceGrid VERSION 1.0 LANGUAGES CXX)

include(GNUInstallDirs)
set(CMAKE_CXX_STANDARD 11)
set(Dina_libs_path /cs/labs/dina/dina/libs)
set(HighFive_path ${Dina_libs_path}/HighFive)
set(CMAKE_PREFIX_PATH ${HighFive_path}/_install/share/HighFive/CMake)
find_package(HighFive 2.0 REQUIRED)
add_subdirectory(cnpy)

FIND_PACKAGE(Boost COMPONENTS program_options REQUIRED)
INCLUDE_DIRECTORIES(${Boost_INCLUDE_DIRS})

SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -O0 -march=native")

set(LIB /cs/labs/dina/punims/gamb)
set(MYLIB /cs/labs/dina/punims/DockingLib)
set(CNPY ${CMAKE_SOURCE_DIR}/cnpy)


file(GLOB SOURCE_FILES_GRID
        GraphInterface.cpp InterfaceGrid.cc MolecularInterface.cc interface.cc
        )

include_directories(${LIB} ${MYLIB} ${CNPY} ${HighFive_path})
link_directories(${LIB} ${MYLIB} ${CNPY} ${HighFive_path})

set(CMAKE_CXX_FLAGS "-lboost_program_options -lboost_system -lboost_filesystem")
add_executable(interface2grid ${SOURCE_FILES_GRID})
target_include_directories(
        interface2grid
        PUBLIC $<TARGET_PROPERTY:HighFive,INTERFACE_INCLUDE_DIRECTORIES>)
target_link_libraries(interface2grid cnpy z dockingLib gamb++ m $<TARGET_PROPERTY:HighFive,INTERFACE_LINK_LIBRARIES>)

#file(GLOB SOURCE_FILES_PC
#        InterfacePointCloud.cc MolecularInterface.cc interface_pc.cc
#        )
#
#add_executable(interface2pointcloud ${SOURCE_FILES_PC})
#target_include_directories(
#        interface2pointcloud
#        PUBLIC $<TARGET_PROPERTY:HighFive,INTERFACE_INCLUDE_DIRECTORIES>)
#target_link_libraries(interface2pointcloud cnpy z dockingLib gamb++ boost m $<TARGET_PROPERTY:HighFive,INTERFACE_LINK_LIBRARIES>)
