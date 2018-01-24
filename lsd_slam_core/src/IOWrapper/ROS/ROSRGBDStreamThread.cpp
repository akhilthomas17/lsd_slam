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
}

void ROSRGBDStreamThread::init()
{
    vid_channel = nh_.resolveName("image");
    depth_channel = nh_.resolveName("depth");

    rgb_sub.subscribe(nh_, vid_channel, 1);
    depth_sub.subscribe(nh_, depth_channel, 1);

    typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::Image, sensor_msgs::Image> sync_pol;
    message_filters::Synchronizer<sync_pol> sync(sync_pol(10), rgb_sub, depth_sub);
    sync.registerCallback(&ROSRGBDStreamThread::rgbdCb, this);
}

void ROSRGBDStreamThread::rgbdCb(const sensor_msgs::ImageConstPtr& rgbMsg,const sensor_msgs::ImageConstPtr& depthMsg)
{
    if(!haveCalib) return;
    cv_bridge::CvImagePtr rgb_ptr = cv_bridge::toCvCopy(rgbMsg, sensor_msgs::image_encodings::RGB8);
    cv_bridge::CvImagePtr depth_ptr = cv_bridge::toCvCopy(depthMsg, sensor_msgs::image_encodings::MONO8);

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
