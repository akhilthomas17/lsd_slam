#include "Tracking/DeepTAMTracker.h"

namespace lsd_slam
{

DeepTAMTracker::DeepTAMTracker(int w, int h, Eigen::Matrix3f K) : SE3Tracker(w, h, K)
{
    client = nh.serviceClient<reinforced_visual_slam::TrackImage>("track_image");
}

SE3 DeepTAMTracker::trackFrameDeepTAM(TrackingReference* reference, Frame* frame, const SE3& frameToReference_initialEstimate)
{
    reinforced_visual_slam::TrackImage srv;
    if (client.call(srv))
    {
        ROS_INFO("Response received");
        return SE3();
    }
    else
    {
        ROS_INFO("No response!!");
        return SE3();
    }
}

}
