# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.16

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/zzh/SLAM/ORB_SLAM2-master

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/zzh/SLAM/ORB_SLAM2-master

# Include any dependencies generated for this target.
include CMakeFiles/myvideo.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/myvideo.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/myvideo.dir/flags.make

CMakeFiles/myvideo.dir/myvideo.cpp.o: CMakeFiles/myvideo.dir/flags.make
CMakeFiles/myvideo.dir/myvideo.cpp.o: myvideo.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/zzh/SLAM/ORB_SLAM2-master/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/myvideo.dir/myvideo.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/myvideo.dir/myvideo.cpp.o -c /home/zzh/SLAM/ORB_SLAM2-master/myvideo.cpp

CMakeFiles/myvideo.dir/myvideo.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/myvideo.dir/myvideo.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/zzh/SLAM/ORB_SLAM2-master/myvideo.cpp > CMakeFiles/myvideo.dir/myvideo.cpp.i

CMakeFiles/myvideo.dir/myvideo.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/myvideo.dir/myvideo.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/zzh/SLAM/ORB_SLAM2-master/myvideo.cpp -o CMakeFiles/myvideo.dir/myvideo.cpp.s

# Object files for target myvideo
myvideo_OBJECTS = \
"CMakeFiles/myvideo.dir/myvideo.cpp.o"

# External object files for target myvideo
myvideo_EXTERNAL_OBJECTS =

Examples/Monocular/myvideo: CMakeFiles/myvideo.dir/myvideo.cpp.o
Examples/Monocular/myvideo: CMakeFiles/myvideo.dir/build.make
Examples/Monocular/myvideo: lib/libORB_SLAM2.so
Examples/Monocular/myvideo: /usr/local/lib/libopencv_gapi.so.4.5.5
Examples/Monocular/myvideo: /usr/local/lib/libopencv_highgui.so.4.5.5
Examples/Monocular/myvideo: /usr/local/lib/libopencv_ml.so.4.5.5
Examples/Monocular/myvideo: /usr/local/lib/libopencv_objdetect.so.4.5.5
Examples/Monocular/myvideo: /usr/local/lib/libopencv_photo.so.4.5.5
Examples/Monocular/myvideo: /usr/local/lib/libopencv_stitching.so.4.5.5
Examples/Monocular/myvideo: /usr/local/lib/libopencv_video.so.4.5.5
Examples/Monocular/myvideo: /usr/local/lib/libopencv_calib3d.so.4.5.5
Examples/Monocular/myvideo: /usr/local/lib/libopencv_dnn.so.4.5.5
Examples/Monocular/myvideo: /usr/local/lib/libopencv_features2d.so.4.5.5
Examples/Monocular/myvideo: /usr/local/lib/libopencv_flann.so.4.5.5
Examples/Monocular/myvideo: /usr/local/lib/libopencv_videoio.so.4.5.5
Examples/Monocular/myvideo: /usr/local/lib/libopencv_imgcodecs.so.4.5.5
Examples/Monocular/myvideo: /usr/local/lib/libopencv_imgproc.so.4.5.5
Examples/Monocular/myvideo: /usr/local/lib/libopencv_core.so.4.5.5
Examples/Monocular/myvideo: /usr/local/lib/libpango_glgeometry.so
Examples/Monocular/myvideo: /usr/local/lib/libpango_geometry.so
Examples/Monocular/myvideo: /usr/local/lib/libpango_plot.so
Examples/Monocular/myvideo: /usr/local/lib/libpango_python.so
Examples/Monocular/myvideo: /usr/local/lib/libpango_scene.so
Examples/Monocular/myvideo: /usr/local/lib/libpango_tools.so
Examples/Monocular/myvideo: /usr/local/lib/libpango_display.so
Examples/Monocular/myvideo: /usr/local/lib/libpango_vars.so
Examples/Monocular/myvideo: /usr/local/lib/libpango_video.so
Examples/Monocular/myvideo: /usr/local/lib/libpango_packetstream.so
Examples/Monocular/myvideo: /usr/local/lib/libpango_windowing.so
Examples/Monocular/myvideo: /usr/local/lib/libpango_opengl.so
Examples/Monocular/myvideo: /usr/local/lib/libpango_image.so
Examples/Monocular/myvideo: /usr/local/lib/libpango_core.so
Examples/Monocular/myvideo: /usr/lib/x86_64-linux-gnu/libGLEW.so
Examples/Monocular/myvideo: /usr/lib/x86_64-linux-gnu/libOpenGL.so
Examples/Monocular/myvideo: /usr/lib/x86_64-linux-gnu/libGLX.so
Examples/Monocular/myvideo: /usr/lib/x86_64-linux-gnu/libGLU.so
Examples/Monocular/myvideo: /usr/local/lib/libtinyobj.so
Examples/Monocular/myvideo: Thirdparty/DBoW2/lib/libDBoW2.so
Examples/Monocular/myvideo: Thirdparty/g2o/lib/libg2o.so
Examples/Monocular/myvideo: CMakeFiles/myvideo.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/zzh/SLAM/ORB_SLAM2-master/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable Examples/Monocular/myvideo"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/myvideo.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/myvideo.dir/build: Examples/Monocular/myvideo

.PHONY : CMakeFiles/myvideo.dir/build

CMakeFiles/myvideo.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/myvideo.dir/cmake_clean.cmake
.PHONY : CMakeFiles/myvideo.dir/clean

CMakeFiles/myvideo.dir/depend:
	cd /home/zzh/SLAM/ORB_SLAM2-master && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/zzh/SLAM/ORB_SLAM2-master /home/zzh/SLAM/ORB_SLAM2-master /home/zzh/SLAM/ORB_SLAM2-master /home/zzh/SLAM/ORB_SLAM2-master /home/zzh/SLAM/ORB_SLAM2-master/CMakeFiles/myvideo.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/myvideo.dir/depend

