#include "Tracking/DeepTAMTracker.h"

namespace lsd_slam
{
DeepTAMTracker::DeepTAMTracker(int w, int h, Eigen::Matrix3f K) : base(w, h, K)
{
}

DeepTAMTracker::~DeepTAMTracker() : ~base()
{
}

SE3 DeepTAMTracker::trackFrameDeepTAM(TrackingReference* reference, Frame* frame, const SE3& frameToReference_initialEstimate)
{

}
