#pragma once

#include "IOWrapper/ROS/ROSImageStreamThread.h"
#include <message_filters/subscriber.h>
#include <message_filters/time_synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>

#include "util/Undistorter.h"

namespace lsd_slam
{

/**
 * RGB and RGBD streams provider using ROS messages.
 */
class ROSRGBDStreamThread : public ROSImageStreamThread
{
public:
ROSRGBDStreamThread();
~ROSRGBDStreamThread();

void init();

/**
 * Callback when RGB images and Depth Images are received.
 */
void rgbdCb(const sensor_msgs::ImageConstPtr &img, const sensor_msgs::ImageConstPtr &depth);
typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::Image, sensor_msgs::Image> sync_pol;

protected:
message_filters::Subscriber<sensor_msgs::Image>* rgb_sub;
message_filters::Subscriber<sensor_msgs::Image>* depth_sub;
message_filters::Synchronizer<sync_pol>* sync;
std::string depth_channel;
};

}
