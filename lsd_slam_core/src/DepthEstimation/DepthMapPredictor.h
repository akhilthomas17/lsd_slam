#pragma once
#include "DepthEstimation/DepthMap.h"
#include "DataStructures/Frame.h"

namespace lsd_slam
{
  class DepthMapPredictor : public DepthMap
  {
  public:
    DepthMapPredictor(int w, int h, const Eigen::Matrix3f& K) : DepthMap(w, h, K){}
    void createKeyframeManager(Frame* new_keyframe);
  };
}
