#include "SlamSystemReinforced.h"

#include "Tracking/DeepTAMTracker.h"

#include "DataStructures/Frame.h"
#include "Tracking/Sim3Tracker.h"
#include "DepthEstimation/DepthMapPredictor.h"
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

#include <reinforced_visual_slam/PredictDepthmap.h>
#include <cv_bridge/cv_bridge.h>
#include <ros/ros.h>

// for mkdir
#include <sys/types.h>
#include <sys/stat.h>

#ifdef ANDROID
#include <android/log.h>
#endif

#include <math.h>

#define PI 3.14159265


using namespace lsd_slam;

SlamSystemReinforced::SlamSystemReinforced(int w, int h, Eigen::Matrix3f K, bool enableSLAM): SlamSystem(w, h, K, enableSLAM)
{
    map =  new DepthMapPredictor(w,h,K);
    tracker = new DeepTAMTracker(w,h,K);
    // Do not use more than 4 levels for odometry tracking
    for (int level = 4; level < PYRAMID_LEVELS; ++level)
        tracker->settings.maxItsPerLvl[level] = 0;
    //thread_mapping = boost::thread(&SlamSystemReinforced::mappingThreadLoop, this);
    printf("Started SlamSystemReinforced\n");

    /*
    printf("Waiting for handshake to complete\n");
    while(!(tracker->shakeHands())){

    }
    printf("Handshake completed\n");
    //*/

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

bool SlamSystemReinforced::getDepthPrediction(const cv::Mat& rgb, cv::Mat& predicted_depth)
{
	ros::NodeHandle nh;
    ros::ServiceClient depthClient = nh.serviceClient<reinforced_visual_slam::PredictDepthmap>("predict_depthmap");
	/** Get the depth prediction from network **/
    //** Call Depth Node **//
    ROS_WARN("Predicting depth using network");
    reinforced_visual_slam::PredictDepthmap srv;
    srv.request.rgb_image = *(cv_bridge::CvImage( std_msgs::Header(),"bgr8",rgb ).toImageMsg());
    if(depthClient.call(srv))
    {
		ROS_INFO("Depth prediction received");

		//** convert srv.response.predicted_depth to cv mat **//
		cv_bridge::CvImagePtr cvImage;
		try
		{
		cvImage = cv_bridge::toCvCopy(srv.response.depthmap, "");
		}
		catch (cv_bridge::Exception& e)
		{
		ROS_ERROR("cv_bridge exception: %s", e.what());
		return false;
		}

		//cv::Mat depthImage = 1.0/cvImage->image;
		predicted_depth = 1.0/cvImage->image;

		//** convert idepth to depth **//

		/*
		float* depth_predicted = reinterpret_cast<float*>(depthImage.data);
		float* idepth_predicted = const reinterpret_cast<float*>(cvImage->image.data);

		
		for(int y=0;y<240;y++)
		{
			for(int x=0;x<320;x++)
			{
				if(!isnanf(idepth_predicted[x+y*width]) && idepth_predicted[x+y*width] > 0)
					depth_predicted[x+y*width] = 1.0 / idepth_predicted[x+y*width];
			}
		}
		*/

		// Network returns predicted idepth of 320x240. Resize it to 640x480
		//cv::resize(depthImage, predicted_depth, predicted_depth.size());
		//depthImage.release();

		//cv::imshow("predicted_depth", rgb);
		//cv::waitKey(0);

		return true;
	}
	else
	{
		return false;
	}
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

    poseConsistencyMutex.lock_shared();
    Sim3 frameToReference_initialEstimate = trackingReferencePose->getCamToWorld().inverse() * keyFrameGraph->allFramePoses.back()->getCamToWorld();
    //SE3 frameToReference_initialEstimate = se3FromSim3(trackingReferencePose->getCamToWorld().inverse()) * se3FromSim3(keyFrameGraph->allFramePoses.back()->getCamToWorld());
    poseConsistencyMutex.unlock_shared();

	struct timeval tv_start, tv_end;
	gettimeofday(&tv_start, NULL);


	/** Using DeepTAM for tracking. The pose update is now independent of LSD SLAM */
	//*
	SE3 newRefToFrame_poseUpdate = tracker->trackFrameDeepTAM(
			trackingReference,
			trackingNewFrame.get(),
			frameToReference_initialEstimate.inverse(), optimizeDeepTAM);

	gettimeofday(&tv_end, NULL);
	msTrackFrame = 0.9*msTrackFrame + 0.1*((tv_end.tv_sec-tv_start.tv_sec)*1000.0f + (tv_end.tv_usec-tv_start.tv_usec)/1000.0f);
	nTrackFrame++;

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


	tracking_lastResidual = tracker->lastResidual;
	tracking_lastUsage = tracker->pointUsage;
	tracking_lastGoodPerBad = tracker->lastGoodCount / (tracker->lastGoodCount + tracker->lastBadCount);
	tracking_lastGoodPerTotal = tracker->lastGoodCount / (trackingNewFrame->width(SE3TRACKING_MIN_LEVEL)*trackingNewFrame->height(SE3TRACKING_MIN_LEVEL));

	
	keyFrameGraph->addFrame(trackingNewFrame.get());

	if (outputWrapper != 0)
	{
		outputWrapper->publishTrackedFrame(trackingNewFrame.get());
	}

	latestTrackedFrame = trackingNewFrame;

	/** KeyFrame selection
	Criterion: maxmimum distance and maximum angle */
	//double maxDist = 0.15;
	//double maxAngle = 5;
	double maxDist = KFDistWeight;
	double maxAngle = KFUsageWeight;

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
		_frameToReference_initialEstimate = SE3();
		printf("New keyframe to be created \n");
	}


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

void SlamSystemReinforced::trackFrameLSD(cv::Mat* rgb, cv::Mat* depth, unsigned int frameID, bool blockUntilMapped, double timestamp)
{	
	cv::Mat grayImg;

	if (rgb->channels() > 1)
		cvtColor(*rgb, grayImg, CV_RGB2GRAY);
	else
		grayImg = *rgb;

	// Create new frame
	std::shared_ptr<Frame> trackingNewFrame(new Frame(frameID, width, height, K, timestamp, grayImg.data));
	trackingNewFrame->setCVImages(rgb->clone(), depth->clone());

	if(!trackingIsGood)
	{
		relocalizer.updateCurrentFrame(trackingNewFrame);

		unmappedTrackedFramesMutex.lock();
		unmappedTrackedFramesSignal.notify_one();
		unmappedTrackedFramesMutex.unlock();
		return;
	}

	currentKeyFrameMutex.lock();
	bool my_createNewKeyframe = createNewKeyFrame;	// pre-save here, to make decision afterwards.
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


	poseConsistencyMutex.lock_shared();
	SE3 frameToReference_initialEstimate = se3FromSim3(
			trackingReferencePose->getCamToWorld().inverse() * keyFrameGraph->allFramePoses.back()->getCamToWorld());
	poseConsistencyMutex.unlock_shared();



	struct timeval tv_start, tv_end;
	gettimeofday(&tv_start, NULL);

	SE3 newRefToFrame_poseUpdate = tracker->trackFrame(
			trackingReference,
			trackingNewFrame.get(),
			frameToReference_initialEstimate);


	gettimeofday(&tv_end, NULL);
	msTrackFrame = 0.9*msTrackFrame + 0.1*((tv_end.tv_sec-tv_start.tv_sec)*1000.0f + (tv_end.tv_usec-tv_start.tv_usec)/1000.0f);
	nTrackFrame++;

	tracking_lastResidual = tracker->lastResidual;
	tracking_lastUsage = tracker->pointUsage;
	tracking_lastGoodPerBad = tracker->lastGoodCount / (tracker->lastGoodCount + tracker->lastBadCount);
	tracking_lastGoodPerTotal = tracker->lastGoodCount / (trackingNewFrame->width(SE3TRACKING_MIN_LEVEL)*trackingNewFrame->height(SE3TRACKING_MIN_LEVEL));


	if(manualTrackingLossIndicated || tracker->diverged || (keyFrameGraph->keyframesAll.size() > INITIALIZATION_PHASE_COUNT && !tracker->trackingWasGood))
	{
		printf("TRACKING LOST for frame %d (%1.2f%% good Points, which is %1.2f%% of available points, %s)!\n",
				trackingNewFrame->id(),
				100*tracking_lastGoodPerTotal,
				100*tracking_lastGoodPerBad,
				tracker->diverged ? "DIVERGED" : "NOT DIVERGED");

		trackingReference->invalidate();

		trackingIsGood = false;
		nextRelocIdx = -1;

		unmappedTrackedFramesMutex.lock();
		unmappedTrackedFramesSignal.notify_one();
		unmappedTrackedFramesMutex.unlock();

		manualTrackingLossIndicated = false;
		return;
	}



	if(plotTracking)
	{
		Eigen::Matrix<float, 20, 1> data;
		data.setZero();
		data[0] = tracker->lastResidual;

		data[3] = tracker->lastGoodCount / (tracker->lastGoodCount + tracker->lastBadCount);
		data[4] = 4*tracker->lastGoodCount / (width*height);
		data[5] = tracker->pointUsage;

		data[6] = tracker->affineEstimation_a;
		data[7] = tracker->affineEstimation_b;
		outputWrapper->publishDebugInfo(data);
	}

	keyFrameGraph->addFrame(trackingNewFrame.get());


	//Sim3 lastTrackedCamToWorld = mostCurrentTrackedFrame->getScaledCamToWorld();//  mostCurrentTrackedFrame->TrackingParent->getScaledCamToWorld() * sim3FromSE3(mostCurrentTrackedFrame->thisToParent_SE3TrackingResult, 1.0);
	if (outputWrapper != 0)
	{
		outputWrapper->publishTrackedFrame(trackingNewFrame.get());
	}


	// Keyframe selection
	latestTrackedFrame = trackingNewFrame;
	if (!my_createNewKeyframe && currentKeyFrame->numMappedOnThisTotal > MIN_NUM_MAPPED)
	{
		Sophus::Vector3d dist = newRefToFrame_poseUpdate.translation() * currentKeyFrame->meanIdepth;
		float minVal = fmin(0.2f + keyFrameGraph->keyframesAll.size() * 0.8f / INITIALIZATION_PHASE_COUNT,1.0f);

		if(keyFrameGraph->keyframesAll.size() < INITIALIZATION_PHASE_COUNT)	minVal *= 0.7;

		lastTrackingClosenessScore = trackableKeyFrameSearch->getRefFrameScore(dist.dot(dist), tracker->pointUsage);

		if (lastTrackingClosenessScore > minVal)
		{
			createNewKeyFrame = true;

			if(enablePrintDebugInfo && printKeyframeSelectionInfo)
				printf("SELECT %d on %d! dist %.3f + usage %.3f = %.3f > 1\n",trackingNewFrame->id(),trackingNewFrame->getTrackingParent()->id(), dist.dot(dist), tracker->pointUsage, trackableKeyFrameSearch->getRefFrameScore(dist.dot(dist), tracker->pointUsage));
		}
		else
		{
			if(enablePrintDebugInfo && printKeyframeSelectionInfo)
				printf("SKIPPD %d on %d! dist %.3f + usage %.3f = %.3f > 1\n",trackingNewFrame->id(),trackingNewFrame->getTrackingParent()->id(), dist.dot(dist), tracker->pointUsage, trackableKeyFrameSearch->getRefFrameScore(dist.dot(dist), tracker->pointUsage));

		}
	}


	unmappedTrackedFramesMutex.lock();
	if(unmappedTrackedFrames.size() < 50 || (unmappedTrackedFrames.size() < 100 && trackingNewFrame->getTrackingParent()->numMappedOnThisTotal < 10))
		unmappedTrackedFrames.push_back(trackingNewFrame);
	unmappedTrackedFramesSignal.notify_one();
	unmappedTrackedFramesMutex.unlock();

	// implement blocking
	if(blockUntilMapped && trackingIsGood)
	{
		boost::unique_lock<boost::mutex> lock(newFrameMappedMutex);
		while(unmappedTrackedFrames.size() > 0)
		{
			//printf("TRACKING IS BLOCKING, waiting for %d frames to finish mapping.\n", (int)unmappedTrackedFrames.size());
			newFrameMappedSignal.wait(lock);
		}
		lock.unlock();
	}
}
