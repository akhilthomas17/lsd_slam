#include "ROSRGBDStreamThread.h"
#include <ros/callback_queue.h>

#include <boost/thread.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "cv_bridge/cv_bridge.h"
#include "util/settings.h"

#include <iostream>
#include <fstream>

namespace lsd_slam
{

using namespace cv;

ROSRGBDStreamThread::ROSRGBDStreamThread()
{
    // Depth image buffer
    depthBuffer = new NotifyBuffer<TimestampedMat>(8);
}

ROSRGBDStreamThread::~ROSRGBDStreamThread()
{
    delete depthBuffer;
    delete rgb_sub;
    delete depth_sub;
    delete sync;
}

void ROSRGBDStreamThread::init()
{
    vid_channel = nh_.resolveName("image");
    depth_channel = nh_.resolveName("depth");

    printf("depth_channel: %s\n", depth_channel.c_str());
    printf("vid_channel: %s\n", vid_channel.c_str());

    rgb_sub = new message_filters::Subscriber<sensor_msgs::Image>(nh_, vid_channel, 10);
    depth_sub = new message_filters::Subscriber<sensor_msgs::Image>(nh_, depth_channel, 10);

    
    sync = new message_filters::Synchronizer<sync_pol> (sync_pol(10), *rgb_sub, *depth_sub);
    sync->registerCallback(boost::bind(&ROSRGBDStreamThread::rgbdCb, this, _1, _2));
}

void ROSRGBDStreamThread::rgbdCb(const sensor_msgs::ImageConstPtr& rgbMsg,const sensor_msgs::ImageConstPtr& depthMsg)
{
    if(!haveCalib) return;
    cv_bridge::CvImagePtr rgb_ptr = cv_bridge::toCvCopy(rgbMsg, sensor_msgs::image_encodings::RGB8);
    ROS_WARN("Input ROS encoding: %s", depthMsg->encoding.c_str());
    cv_bridge::CvImagePtr depth_ptr = cv_bridge::toCvCopy(depthMsg, "");

    if(rgbMsg->header.seq < (unsigned int)lastSEQ)
    {
	printf("Backward-Jump in SEQ detected, but ignoring for now.\n");
	lastSEQ = 0;
	return;
    }
    lastSEQ = rgbMsg->header.seq;

    TimestampedMat bufferItem;
    TimestampedMat bufferItemDepth;

    if(rgbMsg->header.stamp.toSec() != 0)
	bufferItem.timestamp =  Timestamp(rgbMsg->header.stamp.toSec());
    else
	bufferItem.timestamp =  Timestamp(ros::Time::now().toSec());
    bufferItemDepth.timestamp = bufferItem.timestamp;

    if(undistorter != 0)
    {
	assert(undistorter->isValid());
	undistorter->undistort(rgb_ptr->image,bufferItem.data);
    }
    else
    {
	bufferItem.data = rgb_ptr->image;
    }
    bufferItemDepth.data = depth_ptr->image;

    imageBuffer->pushBack(bufferItem);
    depthBuffer->pushBack(bufferItemDepth);
}

}
