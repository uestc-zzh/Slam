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
include CMakeFiles/stereo_euroc.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/stereo_euroc.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/stereo_euroc.dir/flags.make

CMakeFiles/stereo_euroc.dir/Examples/Stereo/stereo_euroc.cc.o: CMakeFiles/stereo_euroc.dir/flags.make
CMakeFiles/stereo_euroc.dir/Examples/Stereo/stereo_euroc.cc.o: Examples/Stereo/stereo_euroc.cc
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/zzh/SLAM/ORB_SLAM2-master/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/stereo_euroc.dir/Examples/Stereo/stereo_euroc.cc.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/stereo_euroc.dir/Examples/Stereo/stereo_euroc.cc.o -c /home/zzh/SLAM/ORB_SLAM2-master/Examples/Stereo/stereo_euroc.cc

CMakeFiles/stereo_euroc.dir/Examples/Stereo/stereo_euroc.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/stereo_euroc.dir/Examples/Stereo/stereo_euroc.cc.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/zzh/SLAM/ORB_SLAM2-master/Examples/Stereo/stereo_euroc.cc > CMakeFiles/stereo_euroc.dir/Examples/Stereo/stereo_euroc.cc.i

CMakeFiles/stereo_euroc.dir/Examples/Stereo/stereo_euroc.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/stereo_euroc.dir/Examples/Stereo/stereo_euroc.cc.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/zzh/SLAM/ORB_SLAM2-master/Examples/Stereo/stereo_euroc.cc -o CMakeFiles/stereo_euroc.dir/Examples/Stereo/stereo_euroc.cc.s

# Object files for target stereo_euroc
stereo_euroc_OBJECTS = \
"CMakeFiles/stereo_euroc.dir/Examples/Stereo/stereo_euroc.cc.o"

# External object files for target stereo_euroc
stereo_euroc_EXTERNAL_OBJECTS =

Examples/Stereo/stereo_euroc: CMakeFiles/stereo_euroc.dir/Examples/Stereo/stereo_euroc.cc.o
Examples/Stereo/stereo_euroc: CMakeFiles/stereo_euroc.dir/build.make
Examples/Stereo/stereo_euroc: lib/libORB_SLAM2.so
Examples/Stereo/stereo_euroc: /usr/local/lib/libopencv_gapi.so.4.5.5
Examples/Stereo/stereo_euroc: /usr/local/lib/libopencv_highgui.so.4.5.5
Examples/Stereo/stereo_euroc: /usr/local/lib/libopencv_ml.so.4.5.5
Examples/Stereo/stereo_euroc: /usr/local/lib/libopencv_objdetect.so.4.5.5
Examples/Stereo/stereo_euroc: /usr/local/lib/libopencv_photo.so.4.5.5
Examples/Stereo/stereo_euroc: /usr/local/lib/libopencv_stitching.so.4.5.5
Examples/Stereo/stereo_euroc: /usr/local/lib/libopencv_video.so.4.5.5
Examples/Stereo/stereo_euroc: /usr/local/lib/libopencv_calib3d.so.4.5.5
Examples/Stereo/stereo_euroc: /usr/local/lib/libopencv_dnn.so.4.5.5
Examples/Stereo/stereo_euroc: /usr/local/lib/libopencv_features2d.so.4.5.5
Examples/Stereo/stereo_euroc: /usr/local/lib/libopencv_flann.so.4.5.5
Examples/Stereo/stereo_euroc: /usr/local/lib/libopencv_videoio.so.4.5.5
Examples/Stereo/stereo_euroc: /usr/local/lib/libopencv_imgcodecs.so.4.5.5
Examples/Stereo/stereo_euroc: /usr/local/lib/libopencv_imgproc.so.4.5.5
Examples/Stereo/stereo_euroc: /usr/local/lib/libopencv_core.so.4.5.5
Examples/Stereo/stereo_euroc: /usr/local/lib/libpango_glgeometry.so
Examples/Stereo/stereo_euroc: /usr/local/lib/libpango_geometry.so
Examples/Stereo/stereo_euroc: /usr/local/lib/libpango_plot.so
Examples/Stereo/stereo_euroc: /usr/local/lib/libpango_python.so
Examples/Stereo/stereo_euroc: /usr/local/lib/libpango_scene.so
Examples/Stereo/stereo_euroc: /usr/local/lib/libpango_tools.so
Examples/Stereo/stereo_euroc: /usr/local/lib/libpango_display.so
Examples/Stereo/stereo_euroc: /usr/local/lib/libpango_vars.so
Examples/Stereo/stereo_euroc: /usr/local/lib/libpango_video.so
Examples/Stereo/stereo_euroc: /usr/local/lib/libpango_packetstream.so
Examples/Stereo/stereo_euroc: /usr/local/lib/libpango_windowing.so
Examples/Stereo/stereo_euroc: /usr/local/lib/libpango_opengl.so
Examples/Stereo/stereo_euroc: /usr/local/lib/libpango_image.so
Examples/Stereo/stereo_euroc: /usr/local/lib/libpango_core.so
Examples/Stereo/stereo_euroc: /usr/lib/x86_64-linux-gnu/libGLEW.so
Examples/Stereo/stereo_euroc: /usr/lib/x86_64-linux-gnu/libOpenGL.so
Examples/Stereo/stereo_euroc: /usr/lib/x86_64-linux-gnu/libGLX.so
Examples/Stereo/stereo_euroc: /usr/lib/x86_64-linux-gnu/libGLU.so
Examples/Stereo/stereo_euroc: /usr/local/lib/libtinyobj.so
Examples/Stereo/stereo_euroc: Thirdparty/DBoW2/lib/libDBoW2.so
Examples/Stereo/stereo_euroc: Thirdparty/g2o/lib/libg2o.so
Examples/Stereo/stereo_euroc: CMakeFiles/stereo_euroc.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/zzh/SLAM/ORB_SLAM2-master/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable Examples/Stereo/stereo_euroc"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/stereo_euroc.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/stereo_euroc.dir/build: Examples/Stereo/stereo_euroc

.PHONY : CMakeFiles/stereo_euroc.dir/build

CMakeFiles/stereo_euroc.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/stereo_euroc.dir/cmake_clean.cmake
.PHONY : CMakeFiles/stereo_euroc.dir/clean

CMakeFiles/stereo_euroc.dir/depend:
	cd /home/zzh/SLAM/ORB_SLAM2-master && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/zzh/SLAM/ORB_SLAM2-master /home/zzh/SLAM/ORB_SLAM2-master /home/zzh/SLAM/ORB_SLAM2-master /home/zzh/SLAM/ORB_SLAM2-master /home/zzh/SLAM/ORB_SLAM2-master/CMakeFiles/stereo_euroc.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/stereo_euroc.dir/depend

