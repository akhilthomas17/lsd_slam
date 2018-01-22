#include "LiveSLAMWrapperReinforced.h"
#include <vector>
#include "util/SophusUtil.h"
#include "SlamSystemReinforced.h"
#include "IOWrapper/ImageDisplay.h"
#include "IOWrapper/Output3DWrapper.h"
#include "IOWrapper/InputImageStream.h"
#include "util/globalFuncs.h"

#include <iostream>

namespace lsd_slam
{


LiveSLAMWrapperReinforced::LiveSLAMWrapperReinforced(InputImageStream* imageStream, Output3DWrapper* outputWrapper):LiveSLAMWrapper(imageStream, outputWrapper)
{
	Sophus::Matrix3f K_sophus;
        K_sophus << fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0;
	// make Odometry
	monoOdometry = new SlamSystemReinforced(width, height, K_sophus, doSlam);
	monoOdometry->setVisualization(outputWrapper);
}


LiveSLAMWrapperReinforced::~LiveSLAMWrapperReinforced()
{
	if(monoOdometry != 0)
		delete monoOdometry;
}

void LiveSLAMWrapperReinforced::newImageCallback(const cv::Mat& img, Timestamp imgTime)
{
	++ imageSeqNumber;

	// Convert image to grayscale, if necessary
	cv::Mat grayImg;
	if (img.channels() == 1)
		grayImg = img;
	else
		cvtColor(img, grayImg, CV_RGB2GRAY);
	

	// Assert that we work with 8 bit images
	assert(grayImg.elemSize() == 1);
	assert(fx != 0 || fy != 0);


	// need to initialize
	if(!isInitialized)
	{
		monoOdometry->randomInit(grayImg.data, imgTime.toSec(), 1);
		isInitialized = true;
	}
	else if(isInitialized && monoOdometry != nullptr)
	{
		monoOdometry->trackFrame(grayImg,imageSeqNumber,false,imgTime.toSec());
	}
    if (monoOdometry->keyFrameChanged)
    {
        logCameraPose(monoOdometry->getLastKeyFramePose(), imgTime.toSec());
        monoOdometry->keyFrameChanged = false;
    }
}

void LiveSLAMWrapperReinforced::resetAll()
{
	if(monoOdometry != nullptr)
	{
		delete monoOdometry;
		printf("Deleted SlamSystem Object!\n");

		Sophus::Matrix3f K;
		K << fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0;
		monoOdometry = new SlamSystemReinforced(width,height,K, doSlam);
		monoOdometry->setVisualization(outputWrapper);

	}
	imageSeqNumber = 0;
	isInitialized = false;

	Util::closeAllWindows();

}

}
