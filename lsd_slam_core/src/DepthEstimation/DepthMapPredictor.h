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
    void fuseDepthMapsManual(Frame* new_keyframe);
    void fuseDepthMapsManual(const float* idepth_predicted, float* idepth_combined);
    void debugPlotsDepthFusion(const float* idepth_predicted, const float* depth_gt);
    ros::NodeHandle nh;
    ros::ServiceClient depthClient;
    /** For Debug plots **/
    cv::Mat debugIdepthPredicted;
    cv::Mat debugIdepthPropagated;
    cv::Mat debugIdepthCombined;
    cv::Mat debugIdepthGt;
  };
}
