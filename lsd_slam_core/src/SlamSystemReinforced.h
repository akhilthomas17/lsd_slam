#pragma once
#include "SlamSystem.h"
#include <opencv2/imgproc/imgproc.hpp>
#include "opencv2/opencv.hpp"



namespace lsd_slam
{

class DeepTAMTracker;

class SlamSystemReinforced : public SlamSystem
{

public:

SlamSystemReinforced(int w, int h, Eigen::Matrix3f K, bool enableSLAM = true);

void trackFrame(cv::Mat* rgb, cv::Mat* depth, unsigned int frameID, bool blockUntilMapped, double timestamp);

void trackFrameTest(cv::Mat* rgb, cv::Mat* depth, unsigned int frameID, bool blockUntilMapped, double timestamp);

void gtDepthInit(cv::Mat* rgb, cv::Mat* depth, double timeStamp, int id);

bool getDepthPrediction(const cv::Mat& rgb, cv::Mat& predicted_depth);

private:

	DeepTAMTracker* tracker;
	SE3 _frameToReference_initialEstimate;

};

}
