# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.20

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/local/APP/jetbrains/clion/2021.2/bin/cmake/linux/bin/cmake

# The command to remove a file.
RM = /usr/local/APP/jetbrains/clion/2021.2/bin/cmake/linux/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /cs/usr/punims/punims/MGClassifier/Interface_grid

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /cs/usr/punims/punims/MGClassifier/Interface_grid

# Include any dependencies generated for this target.
include cnpy/CMakeFiles/example1.dir/depend.make
# Include the progress variables for this target.
include cnpy/CMakeFiles/example1.dir/progress.make

# Include the compile flags for this target's objects.
include cnpy/CMakeFiles/example1.dir/flags.make

cnpy/CMakeFiles/example1.dir/example1.cpp.o: cnpy/CMakeFiles/example1.dir/flags.make
cnpy/CMakeFiles/example1.dir/example1.cpp.o: cnpy/example1.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/cs/usr/punims/punims/MGClassifier/Interface_grid/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object cnpy/CMakeFiles/example1.dir/example1.cpp.o"
	cd /cs/usr/punims/punims/MGClassifier/Interface_grid/cnpy && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/example1.dir/example1.cpp.o -c /cs/usr/punims/punims/MGClassifier/Interface_grid/cnpy/example1.cpp

cnpy/CMakeFiles/example1.dir/example1.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/example1.dir/example1.cpp.i"
	cd /cs/usr/punims/punims/MGClassifier/Interface_grid/cnpy && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /cs/usr/punims/punims/MGClassifier/Interface_grid/cnpy/example1.cpp > CMakeFiles/example1.dir/example1.cpp.i

cnpy/CMakeFiles/example1.dir/example1.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/example1.dir/example1.cpp.s"
	cd /cs/usr/punims/punims/MGClassifier/Interface_grid/cnpy && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /cs/usr/punims/punims/MGClassifier/Interface_grid/cnpy/example1.cpp -o CMakeFiles/example1.dir/example1.cpp.s

# Object files for target example1
example1_OBJECTS = \
"CMakeFiles/example1.dir/example1.cpp.o"

# External object files for target example1
example1_EXTERNAL_OBJECTS =

cnpy/example1: cnpy/CMakeFiles/example1.dir/example1.cpp.o
cnpy/example1: cnpy/CMakeFiles/example1.dir/build.make
cnpy/example1: cnpy/libcnpy.so
cnpy/example1: cnpy/CMakeFiles/example1.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/cs/usr/punims/punims/MGClassifier/Interface_grid/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable example1"
	cd /cs/usr/punims/punims/MGClassifier/Interface_grid/cnpy && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/example1.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
cnpy/CMakeFiles/example1.dir/build: cnpy/example1
.PHONY : cnpy/CMakeFiles/example1.dir/build

cnpy/CMakeFiles/example1.dir/clean:
	cd /cs/usr/punims/punims/MGClassifier/Interface_grid/cnpy && $(CMAKE_COMMAND) -P CMakeFiles/example1.dir/cmake_clean.cmake
.PHONY : cnpy/CMakeFiles/example1.dir/clean

cnpy/CMakeFiles/example1.dir/depend:
	cd /cs/usr/punims/punims/MGClassifier/Interface_grid && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /cs/usr/punims/punims/MGClassifier/Interface_grid /cs/usr/punims/punims/MGClassifier/Interface_grid/cnpy /cs/usr/punims/punims/MGClassifier/Interface_grid /cs/usr/punims/punims/MGClassifier/Interface_grid/cnpy /cs/usr/punims/punims/MGClassifier/Interface_grid/cnpy/CMakeFiles/example1.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : cnpy/CMakeFiles/example1.dir/depend

