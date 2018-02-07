/**
* This file is an extension to LSD-SLAM to combine it with deepTAM.
* Copyright 2018 Akhil Thomas <thomasa at informatik dot uni-freiburg dot de>, University of Freiburg
*
*/
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
	printf("Created LiveSLAMWrapperReinforced\n");
}


LiveSLAMWrapperReinforced::~LiveSLAMWrapperReinforced()
{
	if(monoOdometry != 0)
		delete monoOdometry;
}

void LiveSLAMWrapperReinforced::Loop()
{
	printf("Entering LiveSLAMWrapperReinforced Loop\n");
    while (true) 
    {
    	boost::unique_lock<boost::recursive_mutex> waitLock(imageStream->getBuffer()->getMutex());
        while (!fullResetRequested && !(imageStream->getBuffer()->size() > 0) || !(imageStream->getDepth()->size() > 0)) 
        {
            notifyCondition.wait(waitLock);
        }
        waitLock.unlock();


        if(fullResetRequested)
        {
	        resetAll();
	        fullResetRequested = false;
	        if (!(imageStream->getBuffer()->size() > 0))
	                continue;
        }

        TimestampedMat rgbImage = imageStream->getBuffer()->first();
        imageStream->getBuffer()->popFront();
		TimestampedMat depthImage = imageStream->getDepth()->first();
		imageStream->getDepth()->popFront();
		
		// Scaling the depth Image (DeepTAM needs it in meters)
		//depthImage.data.convertTo(depthImage.data, CV_32FC1, 0.0002);

        // process image
        //Util::displayImage("MyVideo", image.data);
        newImageCallback(rgbImage.data, rgbImage.timestamp, depthImage.data, depthImage.timestamp);
    }
}


void LiveSLAMWrapperReinforced::newImageCallback(cv::Mat& rgbImg, Timestamp imgTime, cv::Mat& depthImg, Timestamp depthTime)
{
	++ imageSeqNumber;
	printf("imageSeqNumber: %d\n", imageSeqNumber);

	cv::Mat grayImg;
	cvtColor(rgbImg, grayImg, CV_RGB2GRAY);


	// Assert that we work with 8 bit images
	assert(grayImg.elemSize() == 1);
	assert(fx != 0 || fy != 0);


	// need to initialize
	if(!isInitialized)
	{
		monoOdometry->gtDepthInit(&rgbImg, &depthImg, imgTime.toSec(), 1);
		isInitialized = true;
	}
	else if(isInitialized && monoOdometry != nullptr)
	{
		monoOdometry->trackFrame(&rgbImg, &depthImg, imageSeqNumber, false, imgTime.toSec());
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
