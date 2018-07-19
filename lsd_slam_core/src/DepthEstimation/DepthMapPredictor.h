#pragma once
#include "DepthEstimation/DepthMap.h"
#include "DataStructures/Frame.h"
#include "util/settings.h"
#include "DepthEstimation/DepthMapPixelHypothesis.h"
#include "util/globalFuncs.h"
#include "IOWrapper/ImageDisplay.h"
#include "GlobalMapping/KeyFrameGraph.h"

#include <stdio.h>
#include <fstream>
#include <iostream>
#include <opencv2/imgproc/imgproc.hpp>

#include <reinforced_visual_slam/DepthFusion.h>
#include <reinforced_visual_slam/PredictDepthmap.h>
#include <ros/ros.h>
#include <cv_bridge/cv_bridge.h>

namespace lsd_slam
{
  class DepthMapPredictor : public DepthMap
  {
  public:
    DepthMapPredictor(int w, int h, const Eigen::Matrix3f& K);
    void createKeyFrame(Frame* new_keyframe);
  private:
    void setFromIdepthMap(const float* idepth_predicted, float* depth_fused, float scale);
    void setFromIdepthMapSparse(const float* idepth_predicted, float* depth_fused, float scale);
    void fuseDepthMapsManual(const float* idepth_predicted, float* idepth_combined);
    void debugPlotsDepthFusion(const float* depth_gt);
    void fillIdepthArray(float* idepth, float* idepthVar);
    ros::NodeHandle nh;
    ros::ServiceClient depthClient, singleImageDepthClient;
    /** For Debug plots **/
    cv::Mat debugIdepthPropagated;
    cv::Mat debugIdepthFused;
    cv::Mat debugIdepthGt;
    bool printDepthPredictionDebugs;
  };
}
