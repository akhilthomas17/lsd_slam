#include "DepthEstimation/DepthMapPredictor.h"

using namespace lsd_slam;

DepthMapPredictor::DepthMapPredictor(int w, int h, const Eigen::Matrix3f& K) : DepthMap(w, h, K)
{
  depthClient = nh.serviceClient<reinforced_visual_slam::PredictDepthmap>("predict_depthmap");
  ROS_INFO("Started DepthMapPredictor");
}

void DepthMapPredictor::createKeyframeManager(Frame* new_keyframe)
{
  if(!predictDepth)
    createKeyFrame(new_keyframe);
  else
    {
    }
}

void DepthMapPredictor::createKeyframePredicted(Frame* new_keyframe)
{
  assert(isValid());
  assert(new_keyframe != nullptr);
  assert(new_keyframe->hasTrackingParent());
  boost::shared_lock<boost::shared_mutex> lock2 = new_keyframe->getActiveLock();

  struct timeval tv_start_all, tv_end_all;
  gettimeofday(&tv_start_all, NULL);

  resetCounters();
  if(plotStereoImages)
  {
    cv::Mat keyFrameImage(new_keyframe->height(), new_keyframe->width(), CV_32F, const_cast<float*>(new_keyframe->image(0)));
    keyFrameImage.convertTo(debugImageHypothesisPropagation, CV_8UC1);
    cv::cvtColor(debugImageHypothesisPropagation, debugImageHypothesisPropagation, CV_GRAY2RGB);
  }

  SE3 oldToNew_SE3 = se3FromSim3(new_keyframe->pose->thisToParent_raw).inverse();
  struct timeval tv_start, tv_end;
  gettimeofday(&tv_start, NULL);

  //** Call Depth Node and get the predicted depth TODO **//
  reinforced_visual_slam::PredictDepthmap srv;
  srv.request.rgb_image = *(cv_bridge::CvImage( std_msgs::Header(),"bgr8",*(new_keyframe->rgbMat()) ).toImageMsg());
  if(depthClient.call(srv)){
    ROS_INFO("Depth prediction received");
    //CV::Mat predicted_depth = convert srv.response.predicted_depth to cv mat
    cv_bridge::CvImagePtr cvImage = cv_bridge::toCvCopy(srv.response.depthmap, "");
    cv::Mat predicted_depth = cvImage->image;
    printf("predicted_depth encoding: %s",cvImage->encoding); 

    gettimeofday(&tv_end, NULL);
    msPropagate = 0.9*msPropagate + 0.1*((tv_end.tv_sec-tv_start.tv_sec)*1000.0f + (tv_end.tv_usec-tv_start.tv_usec)/1000.0f);
    nPropagate++;

    //** Use predicted depth to initialize current depth map TODO **//
    new_keyframe->setDepthFromGroundTruth(reinterpret_cast<float*>(predicted_depth.data));
    initializeFromGTDepth(new_keyframe);

    // make mean inverse depth be one.
    float sumIdepth=0, numIdepth=0;
    for(DepthMapPixelHypothesis* source = currentDepthMap; source < currentDepthMap+width*height; source++)
      {
	if(!source->isValid)
	  continue;
	sumIdepth += source->idepth_smoothed;
	numIdepth++;
      }
    float rescaleFactor = numIdepth / sumIdepth;
    float rescaleFactor2 = rescaleFactor*rescaleFactor;
    for(DepthMapPixelHypothesis* source = currentDepthMap; source < currentDepthMap+width*height; source++)
      {
	if(!source->isValid)
	  continue;
	source->idepth *= rescaleFactor;
	source->idepth_smoothed *= rescaleFactor;
	source->idepth_var *= rescaleFactor2;
	source->idepth_var_smoothed *= rescaleFactor2;
      }
    activeKeyFrame->pose->thisToParent_raw = sim3FromSE3(oldToNew_SE3.inverse(), rescaleFactor);
    activeKeyFrame->pose->invalidateCache();

    // Update depth in keyframe

    gettimeofday(&tv_start, NULL);
    activeKeyFrame->setDepth(currentDepthMap);
    gettimeofday(&tv_end, NULL);
    msSetDepth = 0.9*msSetDepth + 0.1*((tv_end.tv_sec-tv_start.tv_sec)*1000.0f + (tv_end.tv_usec-tv_start.tv_usec)/1000.0f);
    nSetDepth++;

    gettimeofday(&tv_end_all, NULL);
    msCreate = 0.9*msCreate + 0.1*((tv_end_all.tv_sec-tv_start_all.tv_sec)*1000.0f + (tv_end_all.tv_usec-tv_start_all.tv_usec)/1000.0f);
    nCreate++;

    if(plotStereoImages)
      {
	//Util::displayImage( "KeyFramePropagation", debugImageHypothesisPropagation );
      }
  }
}
