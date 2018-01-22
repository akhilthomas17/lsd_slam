#include "SlamSystemReinforced.h"
#include "Tracking/DeepTAMTracker.h"

namespace lsd_slam{

SlamSystemReinforced::SlamSystemReinforced(int w, int h, Eigen::Matrix3f K, bool enableSLAM): SlamSystem(w, h, K, enableSLAM)
{
    tracker = new DeepTAMTracker(w,h,K);
    // Do not use more than 4 levels for odometry tracking
    for (int level = 4; level < PYRAMID_LEVELS; ++level)
        tracker->settings.maxItsPerLvl[level] = 0;
}

void SlamSystemReinforced::trackFrame(cv::Mat& img, unsigned int frameID, bool blockUntilMapped, double timestamp)
{
    trackFrame(img.data, frameID, blockUntilMapped, timestamp);
}

}
