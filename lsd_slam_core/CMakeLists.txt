cmake_minimum_required(VERSION 2.8.7)
project(lsd_slam_core)

# Set the build type. Options are:
#  Coverage : w/ debug symbols, w/o optimization, w/ code-coverage
#  Debug : w/ debug symbols, w/o optimization
#  Release : w/o debug symbols, w/ optimization
#  RelWithDebInfo : w/ debug symbols, w/ optimization
#  MinSizeRel : w/o debug symbols, w/ optimization, stripped binaries
set(CMAKE_BUILD_TYPE Release)

find_package(catkin REQUIRED COMPONENTS
  cv_bridge
  dynamic_reconfigure
  sensor_msgs
  image_transport
  roscpp
  rosbag
  cmake_modules
  lsd_slam_viewer
)

#set (catkin_LIBRARIES "/misc/lmbraid19/thomasa/local/usr/lib/x86_64-linux-gnu/moe/ros-libs/libtinyxml.so/misc/lmbraid19/thomasa/local/usr/lib/x86_64-linux-gnu/moe/ros-libs/libconsole_bridge.so/misc/software/user/ummenhof/ros/kinetic/lib/libcv_bridge.so/misc/software/user/ummenhof/ros/kinetic/lib/libopencv_calib3d3.so.3.2.0/misc/software/user/ummenhof/ros/kinetic/lib/libopencv_core3.so.3.2.0/misc/software/user/ummenhof/ros/kinetic/lib/libopencv_features2d3.so.3.2.0/misc/software/user/ummenhof/ros/kinetic/lib/libopencv_flann3.so.3.2.0/misc/software/user/ummenhof/ros/kinetic/lib/libopencv_highgui3.so.3.2.0/misc/software/user/ummenhof/ros/kinetic/lib/libopencv_imgcodecs3.so.3.2.0/misc/software/user/ummenhof/ros/kinetic/lib/libopencv_imgproc3.so.3.2.0/misc/software/user/ummenhof/ros/kinetic/lib/libopencv_ml3.so.3.2.0/misc/software/user/ummenhof/ros/kinetic/lib/libopencv_objdetect3.so.3.2.0/misc/software/user/ummenhof/ros/kinetic/lib/libopencv_photo3.so.3.2.0/misc/software/user/ummenhof/ros/kinetic/lib/libopencv_shape3.so.3.2.0/misc/software/user/ummenhof/ros/kinetic/lib/libopencv_stitching3.so.3.2.0/misc/software/user/ummenhof/ros/kinetic/lib/libopencv_superres3.so.3.2.0/misc/software/user/ummenhof/ros/kinetic/lib/libopencv_video3.so.3.2.0/misc/software/user/ummenhof/ros/kinetic/lib/libopencv_videoio3.so.3.2.0/misc/software/user/ummenhof/ros/kinetic/lib/libopencv_videostab3.so.3.2.0/misc/software/user/ummenhof/ros/kinetic/lib/libopencv_viz3.so.3.2.0/misc/software/user/ummenhof/ros/kinetic/lib/libopencv_aruco3.so.3.2.0/misc/software/user/ummenhof/ros/kinetic/lib/libopencv_bgsegm3.so.3.2.0/misc/software/user/ummenhof/ros/kinetic/lib/libopencv_bioinspired3.so.3.2.0/misc/software/user/ummenhof/ros/kinetic/lib/libopencv_ccalib3.so.3.2.0/misc/software/user/ummenhof/ros/kinetic/lib/libopencv_cvv3.so.3.2.0/misc/software/user/ummenhof/ros/kinetic/lib/libopencv_datasets3.so.3.2.0/misc/software/user/ummenhof/ros/kinetic/lib/libopencv_dpm3.so.3.2.0/misc/software/user/ummenhof/ros/kinetic/lib/libopencv_face3.so.3.2.0/misc/software/user/ummenhof/ros/kinetic/lib/libopencv_fuzzy3.so.3.2.0/misc/software/user/ummenhof/ros/kinetic/lib/libopencv_hdf3.so.3.2.0/misc/software/user/ummenhof/ros/kinetic/lib/libopencv_line_descriptor3.so.3.2.0/misc/software/user/ummenhof/ros/kinetic/lib/libopencv_optflow3.so.3.2.0/misc/software/user/ummenhof/ros/kinetic/lib/libopencv_phase_unwrapping3.so.3.2.0/misc/software/user/ummenhof/ros/kinetic/lib/libopencv_plot3.so.3.2.0/misc/software/user/ummenhof/ros/kinetic/lib/libopencv_reg3.so.3.2.0/misc/software/user/ummenhof/ros/kinetic/lib/libopencv_rgbd3.so.3.2.0/misc/software/user/ummenhof/ros/kinetic/lib/libopencv_saliency3.so.3.2.0/misc/software/user/ummenhof/ros/kinetic/lib/libopencv_stereo3.so.3.2.0/misc/software/user/ummenhof/ros/kinetic/lib/libopencv_structured_light3.so.3.2.0/misc/software/user/ummenhof/ros/kinetic/lib/libopencv_surface_matching3.so.3.2.0/misc/software/user/ummenhof/ros/kinetic/lib/libopencv_text3.so.3.2.0/misc/software/user/ummenhof/ros/kinetic/lib/libopencv_xfeatures2d3.so.3.2.0/misc/software/user/ummenhof/ros/kinetic/lib/libopencv_ximgproc3.so.3.2.0/misc/software/user/ummenhof/ros/kinetic/lib/libopencv_xobjdetect3.so.3.2.0/misc/software/user/ummenhof/ros/kinetic/lib/libopencv_xphoto3.so.3.2.0/misc/software/user/ummenhof/ros/kinetic/lib/libdynamic_reconfigure_config_init_mutex.so/misc/software/user/ummenhof/ros/kinetic/lib/librosbag.so/misc/software/user/ummenhof/ros/kinetic/lib/librosbag_storage.so/misc/software/user/ummenhof/ros/kinetic/lib/libroslz4.so/misc/lmbraid19/thomasa/local/usr/lib/x86_64-linux-gnu/moe/ros-libs/liblz4.so/misc/software/user/ummenhof/ros/kinetic/lib/libtopic_tools.so/misc/software/user/ummenhof/ros/kinetic/lib/libroscpp.so/usr/lib/x86_64-linux-gnu/libboost_signals.so/misc/software/user/ummenhof/ros/kinetic/lib/librosconsole.so/misc/software/user/ummenhof/ros/kinetic/lib/librosconsole_log4cxx.so/misc/software/user/ummenhof/ros/kinetic/lib/librosconsole_backend_interface.so/misc/lmbraid19/thomasa/local/usr/lib/x86_64-linux-gnu/moe/ros-libs/liblog4cxx.so/usr/lib/x86_64-linux-gnu/libboost_regex.so/misc/software/user/ummenhof/ros/kinetic/lib/libroscpp_serialization.so/misc/software/user/ummenhof/ros/kinetic/lib/librostime.so/misc/software/user/ummenhof/ros/kinetic/lib/libxmlrpcpp.so/misc/software/user/ummenhof/ros/kinetic/lib/libcpp_common.so/usr/lib/x86_64-linux-gnu/libboost_thread.so/usr/lib/x86_64-linux-gnu/libboost_chrono.so/usr/lib/x86_64-linux-gnu/libboost_date_time.so/usr/lib/x86_64-linux-gnu/libboost_atomic.so/usr/lib/x86_64-linux-gnu/libpthread.so/usr/lib/x86_64-linux-gnu/.so/misc/software/user/ummenhof/ros/kinetic/lib/libroslib.so/misc/software/user/ummenhof/ros/kinetic/lib/librospack.so/usr/lib/x86_64-linux-gnu/libpython2.7.so/usr/lib/x86_64-linux-gnu/libboost_filesystem.so/usr/lib/x86_64-linux-gnu/libboost_program_options.so/usr/lib/x86_64-linux-gnu/libboost_system.so/usr/lib/x86_64-linux-gnu/libtinyxml.so")

#set (catkin_LIBRARIES "/misc/lmbraid19/thomasa/local/usr/lib/x86_64-linux-gnu/moe/ros-libs/liblog4cxx.so"${catkin_LIBRARIES})
#set (CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake:${CMAKE_MODULE_PATH}")
#MESSAGE(STATUS "CMAKE_MODULE_PATH_CORE: " ${CMAKE_MODULE_PATH})

find_package(Eigen3 REQUIRED)
find_package(X11 REQUIRED)
#find_package(G2O)
#find_package(SuiteParse)

include(cmake/FindG2O.cmake)
include(cmake/FindSuiteParse.cmake)

message("-- CHOLMOD_INCLUDE_DIR : " ${CHOLMOD_INCLUDE_DIR})
message("-- CSPARSE_INCLUDE_DIR : " ${CSPARSE_INCLUDE_DIR})
message("-- G2O_INCLUDE_DIR : " ${G2O_INCLUDE_DIR})

# FabMap
# uncomment this part to enable fabmap
#add_subdirectory(${PROJECT_SOURCE_DIR}/thirdparty/openFabMap)
#include_directories(${PROJECT_SOURCE_DIR}/thirdparty/openFabMap/include)
#add_definitions("-DHAVE_FABMAP")
#set(FABMAP_LIB openFABMAP )

generate_dynamic_reconfigure_options(
  cfg/LSDDebugParams.cfg
  cfg/LSDParams.cfg
)

catkin_package(
  LIBRARIES lsdslam
  DEPENDS Eigen SuiteSparse
  CATKIN_DEPENDS libg2o 
)

# SSE flags
add_definitions("-DUSE_ROS")
add_definitions("-DENABLE_SSE")

# Also add some useful compiler flag
set(CMAKE_CXX_FLAGS
   "${CMAKE_CXX_FLAGS} -march=native -Wall -std=c++0x"
) 

# Set source files
set(lsd_SOURCE_FILES
  ${PROJECT_SOURCE_DIR}/src/DataStructures/Frame.cpp
  ${PROJECT_SOURCE_DIR}/src/DataStructures/FramePoseStruct.cpp
  ${PROJECT_SOURCE_DIR}/src/DataStructures/FrameMemory.cpp
  ${PROJECT_SOURCE_DIR}/src/SlamSystem.cpp
  ${PROJECT_SOURCE_DIR}/src/LiveSLAMWrapper.cpp
  ${PROJECT_SOURCE_DIR}/src/DepthEstimation/DepthMap.cpp
  ${PROJECT_SOURCE_DIR}/src/DepthEstimation/DepthMapPixelHypothesis.cpp
  ${PROJECT_SOURCE_DIR}/src/util/globalFuncs.cpp
  ${PROJECT_SOURCE_DIR}/src/util/SophusUtil.cpp
  ${PROJECT_SOURCE_DIR}/src/util/settings.cpp
  ${PROJECT_SOURCE_DIR}/src/util/Undistorter.cpp
  ${PROJECT_SOURCE_DIR}/src/Tracking/Sim3Tracker.cpp
  ${PROJECT_SOURCE_DIR}/src/Tracking/Relocalizer.cpp
  ${PROJECT_SOURCE_DIR}/src/Tracking/SE3Tracker.cpp
  ${PROJECT_SOURCE_DIR}/src/Tracking/least_squares.cpp
  ${PROJECT_SOURCE_DIR}/src/Tracking/TrackingReference.cpp
  ${PROJECT_SOURCE_DIR}/src/IOWrapper/Timestamp.cpp
  ${PROJECT_SOURCE_DIR}/src/GlobalMapping/FabMap.cpp
  ${PROJECT_SOURCE_DIR}/src/GlobalMapping/KeyFrameGraph.cpp
  ${PROJECT_SOURCE_DIR}/src/GlobalMapping/g2oTypeSim3Sophus.cpp
  ${PROJECT_SOURCE_DIR}/src/GlobalMapping/TrackableKeyFrameSearch.cpp
)
set(SOURCE_FILES
  ${lsd_SOURCE_FILES}
  ${PROJECT_SOURCE_DIR}/src/IOWrapper/ROS/ROSImageStreamThread.cpp
  ${PROJECT_SOURCE_DIR}/src/IOWrapper/ROS/ROSOutput3DWrapper.cpp
  ${PROJECT_SOURCE_DIR}/src/IOWrapper/OpenCV/ImageDisplay_OpenCV.cpp
)

include_directories(
  include
  ${EIGEN3_INCLUDE_DIR}
  ${PROJECT_SOURCE_DIR}/src
  ${PROJECT_SOURCE_DIR}/thirdparty/Sophus
  ${CSPARSE_INCLUDE_DIR} #Has been set by SuiteParse
  ${CHOLMOD_INCLUDE_DIR} #Has been set by SuiteParse
  ${catkin_INCLUDE_DIRS}
)


# build shared library.
add_library(lsdslam SHARED ${SOURCE_FILES})
target_link_libraries(lsdslam ${FABMAP_LIB} ${G2O_LIBRARIES} ${catkin_LIBRARIES} csparse cxsparse X11)
#rosbuild_link_boost(lsdslam thread)


# build live ros node
add_executable(live_slam src/main_live_odometry.cpp)
add_dependencies(live_slam lsd_slam_viewer_EXPORTED_TARGETS)
target_link_libraries(live_slam lsdslam ${catkin_LIBRARIES} ${G2O_LIBRARIES})


# build image node
add_executable(dataset src/main_on_images.cpp)
target_link_libraries(dataset lsdslam ${catkin_LIBRARIES} ${G2O_LIBRARIES})

# TODO add INSTALL


