#pragma once
#include "SlamSystem.h"
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>


namespace lsd_slam
{

class DeepTAMTracker;
 class DepthMapPredictor;

class SlamSystemReinforced : public SlamSystem
{

public:

SlamSystemReinforced(int w, int h, Eigen::Matrix3f K, bool enableSLAM = true);

void trackFrame(cv::Mat* rgb, cv::Mat* depth, unsigned int frameID, bool blockUntilMapped, double timestamp);

void trackFrameTest(cv::Mat* rgb, cv::Mat* depth, unsigned int frameID, bool blockUntilMapped, double timestamp);

void gtDepthInit(cv::Mat* rgb, cv::Mat* depth, double timeStamp, int id);

 void createNewCurrentKeyframe(std::shared_ptr<Frame> newKeyframeCandidate);

private:

	DeepTAMTracker* tracker;
	SE3 _frameToReference_initialEstimate;
	DepthMapPredictor* map;

};

}
