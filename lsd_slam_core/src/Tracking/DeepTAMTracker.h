/**
* This file is an extension to LSD-SLAM to combine it with deepTAM.
* Copyright 2018 Akhil Thomas <thomasa at informatik dot uni-freiburg dot de>, University of Freiburg
*
*/

#pragma once
#include "Tracking/SE3Tracker.h"
#include <reinforced_visual_slam/TrackImage.h>
#include "ros/ros.h"

namespace lsd_slam
{

class SE3Tracker;

class DeepTAMTracker: public SE3Tracker
{
public:
    DeepTAMTracker(int w, int h, Eigen::Matrix3f K);
    SE3 trackFrameDeepTAM(TrackingReference* reference, Frame* frame, const SE3& frameToReference_initialEstimate);

private:
    ros::NodeHandle nh;
    ros::ServiceClient client;
};

}
