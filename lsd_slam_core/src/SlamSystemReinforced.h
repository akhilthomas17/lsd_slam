#pragma once
#include "SlamSystem.h"

namespace lsd_slam
{

class SlamSystemReinforced : public SlamSystem
{

public:

SlamSystemReinforced(int w, int h, Eigen::Matrix3f K, bool enableSLAM = true);

using SlamSystem::trackFrame;

void trackFrame(cv::Mat& img, unsigned int frameID, bool blockUntilMapped, double timestamp);

};

}
