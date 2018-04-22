#include "DepthEstimation/DepthMapPredictor.h"
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

using namespace lsd_slam;

DepthMapPredictor::DepthMapPredictor(int w, int h, const Eigen::Matrix3f& K) : DepthMap(w, h, K)
{
  depthClient = nh.serviceClient<reinforced_visual_slam::PredictDepthmap>("predict_depthmap");
  debugIdepthPredicted = cv::Mat(h,w, CV_8UC3);
  debugIdepthPropagated = cv::Mat(h,w, CV_8UC3);
  debugIdepthCombined = cv::Mat(h,w, CV_8UC3);
  debugIdepthGt = cv::Mat(h,w, CV_8UC3);
  ROS_INFO("Started DepthMapPredictor");
}

void DepthMapPredictor::createKeyFrame(Frame* new_keyframe)
{
  assert(isValid());
  assert(new_keyframe != nullptr);
  assert(new_keyframe->hasTrackingParent());

  //boost::shared_lock<boost::shared_mutex> lock = activeKeyFrame->getActiveLock();
  boost::shared_lock<boost::shared_mutex> lock2 = new_keyframe->getActiveLock();

  // Store the last keyframe's scale
  double prev_scale = activeKeyFrame->getScaledCamToWorld().scale();

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
  propagateDepth(new_keyframe);
  gettimeofday(&tv_end, NULL);
  msPropagate = 0.9*msPropagate + 0.1*((tv_end.tv_sec-tv_start.tv_sec)*1000.0f + (tv_end.tv_usec-tv_start.tv_usec)/1000.0f);
  nPropagate++;

  activeKeyFrame = new_keyframe;
  activeKeyFramelock = activeKeyFrame->getActiveLock();
  activeKeyFrameImageData = new_keyframe->image(0);
  activeKeyFrameIsReactivated = false;


  gettimeofday(&tv_start, NULL);
  regularizeDepthMap(true, VAL_SUM_MIN_FOR_KEEP);
  gettimeofday(&tv_end, NULL);
  msRegularize = 0.9*msRegularize + 0.1*((tv_end.tv_sec-tv_start.tv_sec)*1000.0f + (tv_end.tv_usec-tv_start.tv_usec)/1000.0f);
  nRegularize++;


  gettimeofday(&tv_start, NULL);
  regularizeDepthMapFillHoles();
  gettimeofday(&tv_end, NULL);
  msFillHoles = 0.9*msFillHoles + 0.1*((tv_end.tv_sec-tv_start.tv_sec)*1000.0f + (tv_end.tv_usec-tv_start.tv_usec)/1000.0f);
  nFillHoles++;


  gettimeofday(&tv_start, NULL);
  regularizeDepthMap(false, VAL_SUM_MIN_FOR_KEEP);
  gettimeofday(&tv_end, NULL);
  msRegularize = 0.9*msRegularize + 0.1*((tv_end.tv_sec-tv_start.tv_sec)*1000.0f + (tv_end.tv_usec-tv_start.tv_usec)/1000.0f);
  nRegularize++;

  printf("predictDepth: %d\n", predictDepth);
  cv::Mat predicted_idepth;

  if(predictDepth){

    /** Temporararilly store the currentDepthMap propagated **/
    memcpy(otherDepthMap,currentDepthMap,width*height*sizeof(DepthMapPixelHypothesis));

    /** Get the depth prediction from network **/
    //** Call Depth Node **//
    ROS_WARN("Predicting depth using network");
    reinforced_visual_slam::PredictDepthmap srv;
    srv.request.rgb_image = *(cv_bridge::CvImage( std_msgs::Header(),"bgr8",*(new_keyframe->rgbMat()) ).toImageMsg());
    if(depthClient.call(srv)){
      ROS_INFO("Depth prediction received");
      //CV::Mat predicted_depth = convert srv.response.predicted_depth to cv mat
      //srv.response.depthmap.step = double(srv.response.depthmap.step);
      ROS_WARN("srv.response.depthmap.step: %d", srv.response.depthmap.step);
      ROS_WARN("srv.response.depthmap.Header: %d", srv.response.depthmap.header.seq);
      ROS_WARN("srv.response.depthmap.encoding: %s", srv.response.depthmap.encoding.c_str());

      cv_bridge::CvImagePtr cvImage;
      
      try
      {
        cvImage = cv_bridge::toCvCopy(srv.response.depthmap, "");
      }
      catch (cv_bridge::Exception& e)
      {
        ROS_ERROR("cv_bridge exception: %s", e.what());
        return;
      }

      // Network returns predicted idepth
      cv::resize(cvImage->image, predicted_idepth, cv::Size(width, height));
      // Multiplying the predicted idepth by the previous depthmap's scale. LSD SLAM assumes idepth to be this way!
      predicted_idepth *= prev_scale;
      printf("predicted_idepth encoding: %d \n",predicted_idepth.type());
      printf("Converted size predicted_idepth rows:%d and cols:%d \n", predicted_idepth.rows, predicted_idepth.cols);

      /** Fuse predicted and projected (*otherDepthMap) idepths to make combined depthmap **/
      // Mat to store combined(predicted + projected) depth
      cv::Mat combined_depth(cv::Size(width, height), CV_32FC1, float(0));
      fuseDepthMapsManual(reinterpret_cast<float*>(predicted_idepth.data), reinterpret_cast<float*>(combined_depth.data));

      //** Sparsify combined depth and update currentDepthMap (Todo) **//
      //new_keyframe->setDepthFromGroundTruth(reinterpret_cast<float*>(combined_depth.data));

      /** Convert the scale of combined_depth back to world (m) and save it inside frame **/
      combined_depth *= prev_scale;
      //activeKeyFrame->setCVDepth(combined_depth);

      /**
      //Normal Plotting for debugs:
      cv::Mat predicted_idepth_plot, combined_depth_plot;
      cv::convertScaleAbs(predicted_idepth * float(255/4), predicted_idepth_plot);
      cv::convertScaleAbs(combined_depth * float(255/4), combined_depth_plot);
      cv::namedWindow( "predicted_idepth", cv::WINDOW_AUTOSIZE ); // Create a window for display.
      cv::imshow( "predicted_idepth", predicted_idepth_plot ); // Show our image inside it.
      cv::namedWindow( "combined_depth", cv::WINDOW_AUTOSIZE ); // Create a window for display.
      cv::imshow( "combined_depth", combined_depth_plot ); // Show our image inside it.
      cv::waitKey(0); // Wait for a keystroke in the window
      **/

    }
    else
      ROS_INFO("No response from depth predictor node!!");
  }

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

  if(predictDepth){
    /** Update the same scale to the propagated idepth for debug plots **/
    for(DepthMapPixelHypothesis* source = otherDepthMap; source < otherDepthMap+width*height; source++)
    {
      if(!source->isValid)
        continue;
      source->idepth *= rescaleFactor;
      source->idepth_smoothed *= rescaleFactor;
      source->idepth_var *= rescaleFactor2;
      source->idepth_var_smoothed *= rescaleFactor2;
    }

    /** Converting the scale of predicted idepth for plotting **/
    predicted_idepth *= rescaleFactor;
    cv::Mat depth_gt = activeKeyFrame->depthMat()->clone();
    depth_gt *= float(1/(rescaleFactor * prev_scale));
    debugPlotsDepthFusion(reinterpret_cast<float*>(predicted_idepth.data), reinterpret_cast<float*>(depth_gt.data));
    Util::displayImage( "iDEPTH Predicted", debugIdepthPredicted, true );
    Util::displayImage( "iDEPTH Propagated", debugIdepthPropagated, true );
    Util::displayImage( "iDEPTH Combined", debugIdepthCombined, true );
    Util::displayImage( "iDEPTH GT", debugIdepthGt, true );

    int waikey = Util::waitKey(0);
  }


  // Update depth in keyframe

  gettimeofday(&tv_start, NULL);
  activeKeyFrame->setDepth(currentDepthMap);
  gettimeofday(&tv_end, NULL);
  msSetDepth = 0.9*msSetDepth + 0.1*((tv_end.tv_sec-tv_start.tv_sec)*1000.0f + (tv_end.tv_usec-tv_start.tv_usec)/1000.0f);
  nSetDepth++;

  gettimeofday(&tv_end_all, NULL);
  msCreate = 0.9*msCreate + 0.1*((tv_end_all.tv_sec-tv_start_all.tv_sec)*1000.0f + (tv_end_all.tv_usec-tv_start_all.tv_usec)/1000.0f);
  nCreate++;

  //if(plotStereoImages)
    //Util::displayImage( "KeyFramePropagation", debugImageHypothesisPropagation );
}

void DepthMapPredictor::fuseDepthMapsManual(const float* idepth_predicted, float* depth_combined){
  /** 
  otherDepthMap is the projected idepthmap, idepth_predicted is the predicted idepthmap. 
  This function updates the idepth and variance of currentDepthMap based on otherDepthMap.
  Also, it computes a new float* depth_combined.
  **/
  const float idepth_var_predicted = VAR_GT_INIT_INITIAL;

  for(int y=0;y<height;y++)
  {
    for(int x=0;x<width;x++)
    {
      // Check if projected values exists for this pixel. If so, merge with predicted values
      if(otherDepthMap[x+y*width].isValid  && otherDepthMap[x+y*width].idepth > 0 && !isnanf(otherDepthMap[x+y*width].idepth))
      {
        // Check if the value predicted for this pixel is valid. If so, merge with projected values
        if(!isnanf(idepth_predicted[x+y*width]) && idepth_predicted[x+y*width] > 0)
        {
          float sumIdepthVar = otherDepthMap[x+y*width].idepth_var + idepth_var_predicted;
          float idepthValue = ( (idepth_predicted[x+y*width] * otherDepthMap[x+y*width].idepth_var) +
                            (otherDepthMap[x+y*width].idepth * idepth_var_predicted) ) / sumIdepthVar;
          float idepthVarValue = (otherDepthMap[x+y*width].idepth_var * idepth_var_predicted) / sumIdepthVar;
          currentDepthMap[x+y*width] = DepthMapPixelHypothesis(
              idepthValue,
              idepthValue,
              idepthVarValue,
              idepthVarValue,
              20);
          depth_combined[x+y*width] = 1.0f/idepthValue;
        }
        // Else, directly use the projected values
        else
        {
          currentDepthMap[x+y*width] = otherDepthMap[x+y*width];
          depth_combined[x+y*width] = 1.0f/otherDepthMap[x+y*width].idepth;
        }
      }
      // Else directly use the predicted depth value if exists
      else
      {
        // Check if the value predicted for this pixel is valid. If so, merge with projected values
        if(!isnanf(idepth_predicted[x+y*width]) && idepth_predicted[x+y*width] > 0)
        {
          currentDepthMap[x+y*width] = DepthMapPixelHypothesis(
              idepth_predicted[x+y*width],
              idepth_predicted[x+y*width],
              idepth_var_predicted,
              idepth_var_predicted,
              20);
          depth_combined[x+y*width] = 1.0f/idepth_predicted[x+y*width];
        }
        // Else, make current value blacklisted and invalid
        else
        {
          currentDepthMap[x+y*width].isValid = false;
          currentDepthMap[x+y*width].blacklisted = 0;
          depth_combined[x+y*width] = 0;
        }
      }
    }
  }
}

void DepthMapPredictor::fuseDepthMapsManual(Frame* new_frame)
{
  /** 
  otherDepthMap is the projected idepthmap, new_keyframe has the predicted idepthmap. 
  This function updates the idepth and variance of currentDepthMap based on otherDepthMap.
  **/
  assert(new_frame->hasIDepthBeenSet());

  activeKeyFramelock = new_frame->getActiveLock();
  activeKeyFrame = new_frame;
  activeKeyFrameImageData = activeKeyFrame->image(0);
  activeKeyFrameIsReactivated = false;

  // Get the predicted idepth and idepth variance
  const float* idepth_predicted = new_frame->idepth();
  const float* idepth_var_predicted = new_frame->idepthVar();

  for(int y=0;y<height;y++)
  {
    for(int x=0;x<width;x++)
    {
      // Check if projected values exists for this pixel. If so, merge with predicted values
      if(otherDepthMap[x+y*width].isValid)
      {
        // Check if the value predicted for this pixel is valid. If so, merge with projected values
        if(!isnanf(idepth_predicted[x+y*width]) && idepth_predicted[x+y*width] > 0)
        {
          float sumIdepthVar = otherDepthMap[x+y*width].idepth_var + idepth_var_predicted[x+y*width];
          float idepthValue = ( (idepth_predicted[x+y*width] * otherDepthMap[x+y*width].idepth_var) +
                            (otherDepthMap[x+y*width].idepth * idepth_var_predicted[x+y*width]) ) / sumIdepthVar;
          float idepthVarValue = (otherDepthMap[x+y*width].idepth_var * idepth_var_predicted[x+y*width]) / sumIdepthVar;
          currentDepthMap[x+y*width] = DepthMapPixelHypothesis(
              idepthValue,
              idepthValue,
              idepthVarValue,
              idepthVarValue,
              20);
        }
        // Else, directly use the projected values
        else
        {
          currentDepthMap[x+y*width] = otherDepthMap[x+y*width];
        }
      }
      // Else directly use the predicted depth value if exists
      else
      {
        // Check if the value predicted for this pixel is valid. If so, merge with projected values
        if(!isnanf(idepth_predicted[x+y*width]) && idepth_predicted[x+y*width] > 0)
        {
          currentDepthMap[x+y*width] = DepthMapPixelHypothesis(
              idepth_predicted[x+y*width],
              idepth_predicted[x+y*width],
              idepth_var_predicted[x+y*width],
              idepth_var_predicted[x+y*width],
              20);
        }
        // Else, make current value blacklisted and invalid
        else
        {
          currentDepthMap[x+y*width].isValid = false;
          currentDepthMap[x+y*width].blacklisted = 0;
        }
      }
    }
  }
  //activeKeyFrame->setDepth(currentDepthMap);
}

void DepthMapPredictor::debugPlotsDepthFusion(const float* idepth_predicted, const float* depth_gt)
{
  if(activeKeyFrame == 0) return;

  cv::Mat keyFrameImage(activeKeyFrame->height(), activeKeyFrame->width(), CV_32F, const_cast<float*>(activeKeyFrameImageData));

  keyFrameImage.convertTo(debugIdepthPredicted, CV_8UC1);
  cv::cvtColor(debugIdepthPredicted, debugIdepthPredicted, CV_GRAY2RGB);
  keyFrameImage.convertTo(debugIdepthPropagated, CV_8UC1);
  cv::cvtColor(debugIdepthPropagated, debugIdepthPropagated, CV_GRAY2RGB);
  keyFrameImage.convertTo(debugIdepthCombined, CV_8UC1);
  cv::cvtColor(debugIdepthCombined, debugIdepthCombined, CV_GRAY2RGB);

  // debug plot & publish sparse version?
  int refID = referenceFrameByID_offset;


  for(int y=0;y<height;y++)
  {
    for(int x=0;x<width;x++)
    {
      int idx = x + y*width;

      if(currentDepthMap[idx].isValid) {
        cv::Vec3b color = currentDepthMap[idx].getVisualizationColor(refID);
        debugIdepthCombined.at<cv::Vec3b>(y,x) = color;
      }

      if(otherDepthMap[idx].isValid) {
        cv::Vec3b color = otherDepthMap[idx].getVisualizationColor(refID);
        debugIdepthPropagated.at<cv::Vec3b>(y,x) = color;
      }

      /** Make colour as per DepthMapPixelHypothesis::getVisualizationColor **/
      float id =  idepth_predicted[idx];
      if(id < 0)
        debugIdepthPredicted.at<cv::Vec3b>(y,x) = cv::Vec3b(255,255,255);
      else if(!isnanf(id))
      {
        // rainbow between 0 and 4
        float r = (0-id) * 255 / 1.0; if(r < 0) r = -r;
        float g = (1-id) * 255 / 1.0; if(g < 0) g = -g;
        float b = (2-id) * 255 / 1.0; if(b < 0) b = -b;

        uchar rc = r < 0 ? 0 : (r > 255 ? 255 : r);
        uchar gc = g < 0 ? 0 : (g > 255 ? 255 : g);
        uchar bc = b < 0 ? 0 : (b > 255 ? 255 : b);

        debugIdepthPredicted.at<cv::Vec3b>(y,x) = cv::Vec3b(255-rc,255-gc,255-bc);
      }

      float d =  depth_gt[idx];
      if(d > 0 && !(isnanf(d)))
      {
        // rainbow between 0 and 4
        float r = (0-float(1/d)) * 255 / 1.0; if(r < 0) r = -r;
        float g = (1-float(1/d)) * 255 / 1.0; if(g < 0) g = -g;
        float b = (2-float(1/d)) * 255 / 1.0; if(b < 0) b = -b;

        uchar rc = r < 0 ? 0 : (r > 255 ? 255 : r);
        uchar gc = g < 0 ? 0 : (g > 255 ? 255 : g);
        uchar bc = b < 0 ? 0 : (b > 255 ? 255 : b);

        debugIdepthGt.at<cv::Vec3b>(y,x) = cv::Vec3b(255-rc,255-gc,255-bc);
      }

    }
  }

}