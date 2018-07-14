/**
* This file is an extension to LSD-SLAM to combine it with deepTAM.
* Copyright 2018 Akhil Thomas <thomasa at informatik dot uni-freiburg dot de>, University of Freiburg
*
*/
#include "Tracking/DeepTAMTracker.h"
namespace lsd_slam
{

#if defined(ENABLE_NEON)
    #define callOptimized(function, arguments) function##NEON arguments
#else
    #if defined(ENABLE_SSE)
        #define callOptimized(function, arguments) (USESSE ? function##SSE arguments : function arguments)
    #else
        #define callOptimized(function, arguments) function arguments
    #endif
#endif

DeepTAMTracker::DeepTAMTracker(int w, int h, Eigen::Matrix3f K) : SE3Tracker(w, h, K)
{
    client = nh.serviceClient<reinforced_visual_slam::TrackImage>("track_image");
    status = nh.serviceClient<reinforced_visual_slam::TrackerStatus>("tracker_status");
    //sub = nh.subscribe("tracker_status", 1, &DeepTAMTracker::statusCallback, this);
    printf("Started DeepTAMTracker\n");
    //trackerStatus = false;
}


bool DeepTAMTracker::shakeHands()
{
    reinforced_visual_slam::TrackerStatus srv;
    if (!client.call(srv))
        return false;
    else
        return srv.response.status.data;

}

SE3 DeepTAMTracker::trackFrameDeepTAM(TrackingReference* reference, Frame* frame,
                                        const Sim3& referenceToFrame_initialEstimate_sim3, bool optimize)
{
    bool debug = true;
    boost::shared_lock<boost::shared_mutex> lock = frame->getActiveLock();
    printf("Tracking current frame \n");
    reinforced_visual_slam::TrackImage srv;
    float current_scale = reference->keyframe->pose->getCamToWorld().scale();
    if (debug){
        ROS_INFO("Type of Depth Image LSD SLAM: %d", reference->keyframe->depthMat()->type());
        ROS_INFO("Type of RGB Image LSD SLAM: %d", reference->keyframe->rgbMat()->type());
        ROS_INFO("Type of RGB current Image LSD SLAM: %d", frame->rgbMat()->type());
        ROS_INFO("Scale of initial estimate guess: %f", current_scale);
    }
    /** Converting the scale of initial guess passed to deepTAM. DeepTAM always expect at scale in m */
    Sim3 scaleInverter;
    scaleInverter.setScale(current_scale);
    SE3 referenceToFrame_initialEstimate = se3FromSim3(scaleInverter * referenceToFrame_initialEstimate_sim3);
    srv.request.keyframe_image = *(cv_bridge::CvImage( std_msgs::Header(),"bgr8",*(reference->keyframe->rgbMat()) ).toImageMsg());
    srv.request.keyframe_depth = *(cv_bridge::CvImage( std_msgs::Header(),"32FC1",*(reference->keyframe->depthMat()) ).toImageMsg());
    srv.request.current_image = *(cv_bridge::CvImage( std_msgs::Header(),"bgr8",*(frame->rgbMat()) ).toImageMsg());
    srv.request.intrinsics = {frame->fx(), frame->fy(), frame->cx(), frame->cy()};

    if (debug)
    {
        cv_bridge::CvImagePtr cvImagergb = cv_bridge::toCvCopy(srv.request.keyframe_image, "");
        ROS_WARN("Test for rgb passed");
        cv_bridge::CvImagePtr cvImagedepth = cv_bridge::toCvCopy(srv.request.keyframe_depth, "");
        ROS_WARN("Test for depth passed");
    }

    srv.request.rotation_prior[0] = referenceToFrame_initialEstimate.so3().log().cast<float>()[0];
    srv.request.rotation_prior[1] = referenceToFrame_initialEstimate.so3().log().cast<float>()[1];
    srv.request.rotation_prior[2] = referenceToFrame_initialEstimate.so3().log().cast<float>()[2];

    srv.request.translation_prior[0] = referenceToFrame_initialEstimate.translation().cast<float>()[0];
    srv.request.translation_prior[1] = referenceToFrame_initialEstimate.translation().cast<float>()[1];
    srv.request.translation_prior[2] = referenceToFrame_initialEstimate.translation().cast<float>()[2];

    printf("Calling DeepTAM client\n");
    SE3 frameToReference;

    if (client.call(srv))
    {
        ROS_INFO("Response received");
        Sophus::Vector3d translation(srv.response.transform.translation.x, srv.response.transform.translation.y, srv.response.transform.translation.z);
        Sophus::Quaterniond orientation;
        orientation.x() = srv.response.transform.rotation.x;
        orientation.y() = srv.response.transform.rotation.y;
        orientation.z() = srv.response.transform.rotation.z;
        orientation.w() = srv.response.transform.rotation.w;

        /** Converting the scale of deepTAM result back to that of current keyframe in LSD SLAM */
        Sim3 frameToReference_unscaled = Sim3(toSophus(orientation), toSophus(translation));
        scaleInverter = Sim3();
        scaleInverter.setScale(1/current_scale);
        frameToReference = se3FromSim3(scaleInverter * frameToReference_unscaled);
        
        if (optimize)
        {
            /** Optimizing DeepTAM tracking using the SE3 tracker */
            frameToReference = trackFrame(reference, frame, frameToReference);
        }
        else
        {
            /** Finding the residuals and updating depth buffers for the found frameToReference */
            //* Uncomment to make DeepTAM tracker work well standalone
            int lvl = SE3TRACKING_MIN_LEVEL;
	    affineEstimation_a = 1; affineEstimation_b = 0;
            Sophus::SE3f referenceToFrame = frameToReference.inverse().cast<float>();
            reference->makePointCloud(lvl);
            callOptimized(calcResidualAndBuffers, (reference->posData[lvl], reference->colorAndVarData[lvl], SE3TRACKING_MIN_LEVEL == lvl ? reference->pointPosInXYGrid[lvl] : 0, reference->numData[lvl], frame, referenceToFrame, lvl, (plotTracking && lvl == SE3TRACKING_MIN_LEVEL)));
            if(useAffineLightningEstimation)
            {
                affineEstimation_a = affineEstimation_a_lastIt;
                affineEstimation_b = affineEstimation_b_lastIt;
            }
	    callOptimized(calcResidualAndBuffers, (reference->posData[lvl], reference->colorAndVarData[lvl], SE3TRACKING_MIN_LEVEL == lvl ? reference->pointPosInXYGrid[lvl] : 0, reference->numData[lvl], frame, referenceToFrame, lvl, (plotTracking && lvl == SE3TRACKING_MIN_LEVEL)));
            lastResidual = callOptimized(calcWeightsAndResidual,(referenceToFrame));
	    ROS_INFO("lastResidual: %f", lastResidual);
            frame->initialTrackedResidual = lastResidual / pointUsage;
            reference->keyframe->numFramesTrackedOnThis++;
            frame->pose->thisToParent_raw = sim3FromSE3(frameToReference,1);
            frame->pose->trackingParent = reference->keyframe->pose;
            //*/
        }
    }
    else
    {
        ROS_INFO("No response from DeepTAM tracker node!!");
    }

    return frameToReference;
}

}
