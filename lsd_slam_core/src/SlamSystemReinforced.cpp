#include "SlamSystemReinforced.h"

#include "Tracking/DeepTAMTracker.h"
#include "Tracking/Relocalizer.h"
#include "util/settings.h"
#include "IOWrapper/Timestamp.h"
#include "opencv2/core/core.hpp"

#include "DataStructures/Frame.h"
#include "Tracking/Sim3Tracker.h"
#include "DepthEstimation/DepthMap.h"
#include "Tracking/TrackingReference.h"
#include "LiveSLAMWrapper.h"
#include "util/globalFuncs.h"
#include "GlobalMapping/KeyFrameGraph.h"
#include "GlobalMapping/TrackableKeyFrameSearch.h"
#include "GlobalMapping/g2oTypeSim3Sophus.h"
#include "IOWrapper/ImageDisplay.h"
#include "IOWrapper/Output3DWrapper.h"
#include <g2o/core/robust_kernel_impl.h>
#include "DataStructures/FrameMemory.h"
#include "deque"

// for mkdir
#include <sys/types.h>
#include <sys/stat.h>

#ifdef ANDROID
#include <android/log.h>
#endif

#include "opencv2/opencv.hpp"
#include <math.h>

#define PI 3.14159265


using namespace lsd_slam;

SlamSystemReinforced::SlamSystemReinforced(int w, int h, Eigen::Matrix3f K, bool enableSLAM): SlamSystem(w, h, K, enableSLAM)
{
    tracker = new DeepTAMTracker(w,h,K);
    // Do not use more than 4 levels for odometry tracking
    for (int level = 4; level < PYRAMID_LEVELS; ++level)
        tracker->settings.maxItsPerLvl[level] = 0;
    printf("Started SlamSystemReinforced\n");
}


void SlamSystemReinforced::gtDepthInit(cv::Mat* rgb, cv::Mat* depth, double timeStamp, int id)
{
    printf("Doing GT initialization!\n");

    cv::Mat grayImg;
    if (rgb->channels() > 1)
    	cvtColor(*rgb, grayImg, CV_RGB2GRAY);
    else
    	grayImg = *rgb;

	// Scaling the depth Image (If input is in millimeters)
	// cv::Mat depthImg;
	//depth->convertTo(depthImg, CV_32FC1, 0.001);

	currentKeyFrameMutex.lock();

	currentKeyFrame.reset(new Frame(id, width, height, K, timeStamp, grayImg.data));
	currentKeyFrame->setDepthFromGroundTruth(reinterpret_cast<float*>(depth->data));
	// Adding CV Mat rgb and depth image pointers to the saved Frame
	currentKeyFrame->setCVImages(rgb->clone(), depth->clone());

	map->initializeFromGTDepth(currentKeyFrame.get());
	keyFrameGraph->addFrame(currentKeyFrame.get());

	currentKeyFrameMutex.unlock();

	if(doSlam)
	{
		keyFrameGraph->idToKeyFrameMutex.lock();
		keyFrameGraph->idToKeyFrame.insert(std::make_pair(currentKeyFrame->id(), currentKeyFrame));
		keyFrameGraph->idToKeyFrameMutex.unlock();
	}
	if(continuousPCOutput && outputWrapper != 0) outputWrapper->publishKeyframe(currentKeyFrame.get());

	printf("Done GT initialization!\n");
}


void SlamSystemReinforced::trackFrame(cv::Mat* rgb, cv::Mat* depth, unsigned int frameID, bool blockUntilMapped, double timestamp)
{
	cv::Mat grayImg;

	if (rgb->channels() > 1)
		cvtColor(*rgb, grayImg, CV_RGB2GRAY);
	else
		grayImg = *rgb;

	// Create new frame
	std::shared_ptr<Frame> trackingNewFrame(new Frame(frameID, width, height, K, timestamp, grayImg.data));
	trackingNewFrame->setCVImages(rgb->clone(), depth->clone());

	/** Relocalization
	if(!trackingIsGood)
	{
		relocalizer.updateCurrentFrame(trackingNewFrame);

		unmappedTrackedFramesMutex.lock();
		unmappedTrackedFramesSignal.notify_one();
		unmappedTrackedFramesMutex.unlock();
		return;
	}
	*/

	currentKeyFrameMutex.lock();
	if(trackingReference->keyframe != currentKeyFrame.get() || currentKeyFrame->depthHasBeenUpdatedFlag)
	{
		trackingReference->importFrame(currentKeyFrame.get());
		currentKeyFrame->depthHasBeenUpdatedFlag = false;
		trackingReferenceFrameSharedPT = currentKeyFrame;
	}

	FramePoseStruct* trackingReferencePose = trackingReference->keyframe->pose;
	currentKeyFrameMutex.unlock();

	// DO TRACKING & Show tracking result.
	if(enablePrintDebugInfo && printThreadingInfo)
		printf("TRACKING %d on %d\n", trackingNewFrame->id(), trackingReferencePose->frameID);

	//printf("Trial KF pose\n");
	//Sim3 dummySim3 = trackingReferencePose->getCamToWorld().inverse();
	//printf("Trial Last pose\n");
	//dummySim3 = keyFrameGraph->allFramePoses.back()->getCamToWorld();


	poseConsistencyMutex.lock_shared();
	SE3 frameToReference_initialEstimate = se3FromSim3(
			trackingReferencePose->getCamToWorld().inverse() * keyFrameGraph->allFramePoses.back()->getCamToWorld());
	poseConsistencyMutex.unlock_shared();



	struct timeval tv_start, tv_end;
	gettimeofday(&tv_start, NULL);


	/** Just for testing purposes so that the system won't break in the beginning */
	/**
	SE3 dummy_poseUpdate = tracker->trackFrame(
			trackingReference,
			trackingNewFrame.get(),
			frameToReference_initialEstimate);
	//*/
        //for(int i=0; i<10; i++)
	/** Using DeepTAM for tracking. The pose update is now independent of LSD SLAM */
	SE3 newRefToFrame_poseUpdate = tracker->trackFrameDeepTAM(
			trackingReference,
			trackingNewFrame.get(),
			frameToReference_initialEstimate);

	//printf("Response from DeepTAM tracker\n");

	/**

	Sophus::Quaternionf quat = dummy_poseUpdate.unit_quaternion().cast<float>();
        Eigen::Vector3f trans = dummy_poseUpdate.translation().cast<float>();

        printf("SE3Tracker: %f %f %f %f %f %f %f\n",
                        trans[0],
                        trans[1],
                        trans[2],
                        quat.x(),
                        quat.y(),
                        quat.z(),
                        quat.w());

    */

	Sophus::Quaternionf quat = newRefToFrame_poseUpdate.unit_quaternion().cast<float>();
	Eigen::Vector3f trans = newRefToFrame_poseUpdate.translation().cast<float>();

	printf("DeepTAMTracker: %f %f %f %f %f %f %f\n",
                        trans[0],
                        trans[1],
                        trans[2],
                        quat.x(),
                        quat.y(),
                        quat.z(),
                        quat.w());

	gettimeofday(&tv_end, NULL);
	msTrackFrame = 0.9*msTrackFrame + 0.1*((tv_end.tv_sec-tv_start.tv_sec)*1000.0f + (tv_end.tv_usec-tv_start.tv_usec)/1000.0f);
	nTrackFrame++;
	
	keyFrameGraph->addFrame(trackingNewFrame.get());

	//Sim3 lastTrackedCamToWorld = mostCurrentTrackedFrame->getScaledCamToWorld();//  mostCurrentTrackedFrame->TrackingParent->getScaledCamToWorld() * sim3FromSE3(mostCurrentTrackedFrame->thisToParent_SE3TrackingResult, 1.0);
	if (outputWrapper != 0)
	{
		outputWrapper->publishTrackedFrame(trackingNewFrame.get());
	}

	/** KeyFrame selection
	Criterion: maxmimum distance and maximum angle */

	double maxDist = 0.15;
	double maxAngle = 5;

	Sophus::Vector3d distVec = newRefToFrame_poseUpdate.translation();
	double distSquare = distVec.dot(distVec);
	//printf("Calculating angle from unit_quaternion\n");
	double angle = 2 * acos (newRefToFrame_poseUpdate.unit_quaternion().w()) * 180.0 / PI;
	//float angle;
	//newRefToFrame_poseUpdate.so3().logAndTheta(*this, angle);// 180.0 / PI;
	printf("distSquare: %f\n",distSquare );
	printf("angle: %f\n",angle );

	if ((distSquare > maxDist*maxDist) || (angle > maxAngle))
	{
		createNewKeyFrame = true;
		printf("New keyframe to be created \n");
	}

	latestTrackedFrame = trackingNewFrame;

	unmappedTrackedFramesMutex.lock();
	if(unmappedTrackedFrames.size() < 50 || (unmappedTrackedFrames.size() < 100 && trackingNewFrame->getTrackingParent()->numMappedOnThisTotal < 10))
		unmappedTrackedFrames.push_back(trackingNewFrame);
	unmappedTrackedFramesSignal.notify_one();
	unmappedTrackedFramesMutex.unlock();

	// implement blocking
	if(blockUntilMapped)
	{
		boost::unique_lock<boost::mutex> lock(newFrameMappedMutex);
		while(unmappedTrackedFrames.size() > 0)
		{
			printf("TRACKING IS BLOCKING, waiting for %d frames to finish mapping.\n", (int)unmappedTrackedFrames.size());
			newFrameMappedSignal.wait(lock);
		}
		lock.unlock();
	}
}
