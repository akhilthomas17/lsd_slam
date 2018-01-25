/**
* This file is an extension to LSD-SLAM to combine it with deepTAM.
* Copyright 2018 Akhil Thomas <thomasa at informatik dot uni-freiburg dot de>, University of Freiburg
*
*/
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
    
    srv.request.keyframe_image = *(cv_bridge::CvImage( std_msgs::Header(),"bgr8",*(reference->keyframe->rgbMat()) ).toImageMsg());
    srv.request.keyframe_depth = *(cv_bridge::CvImage( std_msgs::Header(),"mono8",*(reference->keyframe->depthMat()) ).toImageMsg());
    srv.request.current_image = *(cv_bridge::CvImage( std_msgs::Header(),"bgr8",*(frame->rgbMat()) ).toImageMsg());
    srv.request.intrinsics = {frame->fx(), frame->fy(), frame->cx(), frame->cy()};

    srv.request.rotation_prior[0] = frameToReference_initialEstimate.so3().log().cast<float>()[0];
    srv.request.rotation_prior[1] = frameToReference_initialEstimate.so3().log().cast<float>()[1];
    srv.request.rotation_prior[2] = frameToReference_initialEstimate.so3().log().cast<float>()[2];

    srv.request.translation_prior[0] = frameToReference_initialEstimate.translation().cast<float>()[0];
    srv.request.translation_prior[1] = frameToReference_initialEstimate.translation().cast<float>()[1];
    srv.request.translation_prior[2] = frameToReference_initialEstimate.translation().cast<float>()[2];

    if (client.call(srv))
    {
        ROS_INFO("Response received");
        Sophus::Vector3d translation(srv.response.transform.translation.x, srv.response.transform.translation.y, srv.response.transform.translation.z);
        Sophus::Quaterniond orientation(srv.response.transform.rotation.x, srv.response.transform.rotation.y, srv.response.transform.rotation.z, srv.response.transform.rotation.w);
        return SE3(toSophus(orientation), toSophus(translation));
    }
    else
    {
        ROS_INFO("No response!!");
        return SE3();
    }
}

}
