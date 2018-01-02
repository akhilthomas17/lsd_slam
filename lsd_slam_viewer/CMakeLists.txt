cmake_minimum_required(VERSION 2.4.6)
project(lsd_slam_viewer)

# Set the build type. Options are:
#  Coverage : w/ debug symbols, w/o optimization, w/ code-coverage
#  Debug : w/ debug symbols, w/o optimization
#  Release : w/o debug symbols, w/ optimization
#  RelWithDebInfo : w/ debug symbols, w/ optimization
#  MinSizeRel : w/o debug symbols, w/ optimization, stripped binaries
set(CMAKE_BUILD_TYPE Release)

ADD_SUBDIRECTORY(${PROJECT_SOURCE_DIR}/thirdparty/Sophus)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${PROJECT_SOURCE_DIR}/cmake)
#MESSAGE(STATUS "CMAKE_MODULE_PATH:	"${CMAKE_MODULE_PATH}) 

find_package(catkin REQUIRED COMPONENTS
  cv_bridge
  dynamic_reconfigure
  sensor_msgs
  roscpp
  rosbag
  message_generation
  roslib
  cmake_modules
)

#set (catkin_LIBRARIES "/misc/lmbraid19/thomasa/local/usr/lib/x86_64-linux-gnu/moe/ros-libs/libtinyxml.so/misc/lmbraid19/thomasa/local/usr/lib/x86_64-linux-gnu/moe/ros-libs/libconsole_bridge.so/misc/software/user/ummenhof/ros/kinetic/lib/libcv_bridge.so/misc/software/user/ummenhof/ros/kinetic/lib/libopencv_calib3d3.so.3.2.0/misc/software/user/ummenhof/ros/kinetic/lib/libopencv_core3.so.3.2.0/misc/software/user/ummenhof/ros/kinetic/lib/libopencv_features2d3.so.3.2.0/misc/software/user/ummenhof/ros/kinetic/lib/libopencv_flann3.so.3.2.0/misc/software/user/ummenhof/ros/kinetic/lib/libopencv_highgui3.so.3.2.0/misc/software/user/ummenhof/ros/kinetic/lib/libopencv_imgcodecs3.so.3.2.0/misc/software/user/ummenhof/ros/kinetic/lib/libopencv_imgproc3.so.3.2.0/misc/software/user/ummenhof/ros/kinetic/lib/libopencv_ml3.so.3.2.0/misc/software/user/ummenhof/ros/kinetic/lib/libopencv_objdetect3.so.3.2.0/misc/software/user/ummenhof/ros/kinetic/lib/libopencv_photo3.so.3.2.0/misc/software/user/ummenhof/ros/kinetic/lib/libopencv_shape3.so.3.2.0/misc/software/user/ummenhof/ros/kinetic/lib/libopencv_stitching3.so.3.2.0/misc/software/user/ummenhof/ros/kinetic/lib/libopencv_superres3.so.3.2.0/misc/software/user/ummenhof/ros/kinetic/lib/libopencv_video3.so.3.2.0/misc/software/user/ummenhof/ros/kinetic/lib/libopencv_videoio3.so.3.2.0/misc/software/user/ummenhof/ros/kinetic/lib/libopencv_videostab3.so.3.2.0/misc/software/user/ummenhof/ros/kinetic/lib/libopencv_viz3.so.3.2.0/misc/software/user/ummenhof/ros/kinetic/lib/libopencv_aruco3.so.3.2.0/misc/software/user/ummenhof/ros/kinetic/lib/libopencv_bgsegm3.so.3.2.0/misc/software/user/ummenhof/ros/kinetic/lib/libopencv_bioinspired3.so.3.2.0/misc/software/user/ummenhof/ros/kinetic/lib/libopencv_ccalib3.so.3.2.0/misc/software/user/ummenhof/ros/kinetic/lib/libopencv_cvv3.so.3.2.0/misc/software/user/ummenhof/ros/kinetic/lib/libopencv_datasets3.so.3.2.0/misc/software/user/ummenhof/ros/kinetic/lib/libopencv_dpm3.so.3.2.0/misc/software/user/ummenhof/ros/kinetic/lib/libopencv_face3.so.3.2.0/misc/software/user/ummenhof/ros/kinetic/lib/libopencv_fuzzy3.so.3.2.0/misc/software/user/ummenhof/ros/kinetic/lib/libopencv_hdf3.so.3.2.0/misc/software/user/ummenhof/ros/kinetic/lib/libopencv_line_descriptor3.so.3.2.0/misc/software/user/ummenhof/ros/kinetic/lib/libopencv_optflow3.so.3.2.0/misc/software/user/ummenhof/ros/kinetic/lib/libopencv_phase_unwrapping3.so.3.2.0/misc/software/user/ummenhof/ros/kinetic/lib/libopencv_plot3.so.3.2.0/misc/software/user/ummenhof/ros/kinetic/lib/libopencv_reg3.so.3.2.0/misc/software/user/ummenhof/ros/kinetic/lib/libopencv_rgbd3.so.3.2.0/misc/software/user/ummenhof/ros/kinetic/lib/libopencv_saliency3.so.3.2.0/misc/software/user/ummenhof/ros/kinetic/lib/libopencv_stereo3.so.3.2.0/misc/software/user/ummenhof/ros/kinetic/lib/libopencv_structured_light3.so.3.2.0/misc/software/user/ummenhof/ros/kinetic/lib/libopencv_surface_matching3.so.3.2.0/misc/software/user/ummenhof/ros/kinetic/lib/libopencv_text3.so.3.2.0/misc/software/user/ummenhof/ros/kinetic/lib/libopencv_xfeatures2d3.so.3.2.0/misc/software/user/ummenhof/ros/kinetic/lib/libopencv_ximgproc3.so.3.2.0/misc/software/user/ummenhof/ros/kinetic/lib/libopencv_xobjdetect3.so.3.2.0/misc/software/user/ummenhof/ros/kinetic/lib/libopencv_xphoto3.so.3.2.0/misc/software/user/ummenhof/ros/kinetic/lib/libdynamic_reconfigure_config_init_mutex.so/misc/software/user/ummenhof/ros/kinetic/lib/librosbag.so/misc/software/user/ummenhof/ros/kinetic/lib/librosbag_storage.so/misc/software/user/ummenhof/ros/kinetic/lib/libroslz4.so/misc/lmbraid19/thomasa/local/usr/lib/x86_64-linux-gnu/moe/ros-libs/liblz4.so/misc/software/user/ummenhof/ros/kinetic/lib/libtopic_tools.so/misc/software/user/ummenhof/ros/kinetic/lib/libroscpp.so/usr/lib/x86_64-linux-gnu/libboost_signals.so/misc/software/user/ummenhof/ros/kinetic/lib/librosconsole.so/misc/software/user/ummenhof/ros/kinetic/lib/librosconsole_log4cxx.so/misc/software/user/ummenhof/ros/kinetic/lib/librosconsole_backend_interface.so/misc/lmbraid19/thomasa/local/usr/lib/x86_64-linux-gnu/moe/ros-libs/liblog4cxx.so/usr/lib/x86_64-linux-gnu/libboost_regex.so/misc/software/user/ummenhof/ros/kinetic/lib/libroscpp_serialization.so/misc/software/user/ummenhof/ros/kinetic/lib/librostime.so/misc/software/user/ummenhof/ros/kinetic/lib/libxmlrpcpp.so/misc/software/user/ummenhof/ros/kinetic/lib/libcpp_common.so/usr/lib/x86_64-linux-gnu/libboost_thread.so/usr/lib/x86_64-linux-gnu/libboost_chrono.so/usr/lib/x86_64-linux-gnu/libboost_date_time.so/usr/lib/x86_64-linux-gnu/libboost_atomic.so/usr/lib/x86_64-linux-gnu/libpthread.so/usr/lib/x86_64-linux-gnu/.so/misc/software/user/ummenhof/ros/kinetic/lib/libroslib.so/misc/software/user/ummenhof/ros/kinetic/lib/librospack.so/usr/lib/x86_64-linux-gnu/libpython2.7.so/usr/lib/x86_64-linux-gnu/libboost_filesystem.so/usr/lib/x86_64-linux-gnu/libboost_program_options.so/usr/lib/x86_64-linux-gnu/libboost_system.so/usr/lib/x86_64-linux-gnu/libtinyxml.so")

find_package(OpenGL REQUIRED)
set(QT_USE_QTOPENGL TRUE)
set(QT_USE_QTXML TRUE)
find_package(QGLViewer REQUIRED)
find_package(Eigen REQUIRED)
#include(cmake/FindEigen3.cmake)
find_package(OpenCV REQUIRED)
find_package(Boost REQUIRED COMPONENTS thread)

set(QGLVIEWER_INCLUDE_DIR /misc/lmbraid19/thomasa/local/usr/local/include/QGLViewer)

set(QGLVIEWER_LIBRARY /misc/lmbraid19/thomasa/local/usr/local/lib/libQGLViewer.so)


MESSAGE( STATUS "QGLVIEWER_INCLUDE_DIR:         " ${QGLVIEWER_INCLUDE_DIR} )
MESSAGE( STATUS "QT_INCLUDES:         " ${QT_INCLUDES} )
MESSAGE( STATUS "catkin_INCLUDE_DIRS:         " ${catkin_INCLUDE_DIRS} )
MESSAGE( STATUS "EIGEN_INCLUDE_DIR:         " ${EIGEN_INCLUDE_DIR} )

include_directories(${QGLVIEWER_INCLUDE_DIR}
		    ${catkin_INCLUDE_DIRS} 
		    ${EIGEN_INCLUDE_DIR}
		    ${QT_INCLUDES} )

# SSE flags
set(CMAKE_CXX_FLAGS
   "${CMAKE_CXX_FLAGS} -march=native -Wall -std=c++0x"
)

add_message_files(DIRECTORY msg FILES keyframeMsg.msg keyframeGraphMsg.msg)
generate_messages(
DEPENDENCIES
std_msgs
)

generate_dynamic_reconfigure_options(
  cfg/LSDSLAMViewerParams.cfg
)

catkin_package()

# Sources files
set(SOURCE_FILES         
  src/PointCloudViewer.cpp
  src/KeyFrameDisplay.cpp
  src/KeyFrameGraphDisplay.cpp
  src/settings.cpp
)

set(HEADER_FILES     
  src/PointCloudViewer.h
  src/KeyFrameDisplay.h
  src/KeyFrameGraphDisplay.h
  src/settings.h
)

include_directories(
  ${PROJECT_SOURCE_DIR}/thirdparty/Sophus
)  

MESSAGE( STATUS "QGLViewer_LIBRARIES:         " ${QGLViewer_LIBRARIES} )
MESSAGE( STATUS "QGLVIEWER_LIBRARY:         " ${QGLVIEWER_LIBRARY} )
MESSAGE( STATUS "QT_LIBRARIES:         " ${QT_LIBRARIES} )
MESSAGE( STATUS "catkin_LIBRARIES:   "${catkin_LIBRARIES})

add_executable(viewer src/main_viewer.cpp ${SOURCE_FILES} ${HEADER_FILES})
add_dependencies(viewer ${catkin_EXPORTED_TARGETS})

target_link_libraries(viewer ${QGLViewer_LIBRARIES}
			     ${QGLVIEWER_LIBRARY} 
			     ${catkin_LIBRARIES}
			     ${Boost_LIBRARIES}
			     ${QT_LIBRARIES}
			     GL glut GLU
		     )

#add_executable(videoStitch src/main_stitchVideos.cpp)
#target_link_libraries(viewer ${QGLViewer_LIBRARIES}
#			     ${QGLVIEWER_LIBRARY}
#			     ${catkin_LIBRARIES}
#			     ${QT_LIBRARIES}
#			     GL glut GLU
#		      )

