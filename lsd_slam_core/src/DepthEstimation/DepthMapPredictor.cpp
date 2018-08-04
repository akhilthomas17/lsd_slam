#include "DepthEstimation/DepthMapPredictor.h"
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <half.hpp>

using namespace lsd_slam;
using half_float::half;


DepthMapPredictor::DepthMapPredictor(int w, int h, const Eigen::Matrix3f& K) : DepthMap(w, h, K)
{
  depthClient = nh.serviceClient<reinforced_visual_slam::DepthFusion>("fuse_depthmap");
  singleImageDepthClient = nh.serviceClient<reinforced_visual_slam::PredictDepthmap>("predict_depthmap");
  debugIdepthPropagated = cv::Mat(h,w, CV_8UC3);
  debugIdepthFused = cv::Mat(h,w, CV_8UC3);
  debugIdepthGt = cv::Mat(h,w, CV_8UC3);
  printDepthPredictionDebugs = false;
  if(predictDepth){
    std::string service;
    if(depthCompletion)
      service = "fuse_depthmap";
    else
      service = "predict_depthmap";
    while(!ros::service::exists(service, false)){
        sleep(2);
        printf("Waiting for service %s to start..\n", service.c_str());
    }
  }
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

  if(doSlam && !useGtDepth){

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

    /** Save images to file for depth training **/
    if (writeDepthToFile)
    {
      activeKeyFrame->setDepth(currentDepthMap);
      float scale = activeKeyFrame->getScaledCamToWorld().scale();
      cv::Mat* depthGt = activeKeyFrame->depthGTMat();
      //depthGt->convertTo(*depthGt, CV_16U, 5000); This was not needed inside pure LSD SLAM
      cv::Mat* rgb = activeKeyFrame->rgbMat();

      // De-scale the idepth and idepthVar stored back to world (m)
      const float* idepthFloat = activeKeyFrame->idepth();
      const float* idepthVarFloat = activeKeyFrame->idepthVar();
      int size = width*height;

      half idepthData[size];
      half idepthVarData[size];

      float maxDepth = 0;
      float maxDepthVar = 0;

      for (int ii=0; ii < size; ii++)
      {
        if (*(idepthFloat + ii) > 0 && *(idepthVarFloat + ii)>0)
        {
          *(idepthData+ii) = *(idepthFloat+ii)/scale;
          *(idepthVarData+ii) = *(idepthVarFloat+ii)/scale;
          if(*(idepthData+ii) > maxDepth)
            maxDepth = *(idepthData+ii);
          if(*(idepthVarData+ii) > maxDepthVar)
            maxDepthVar = *(idepthVarData+ii);
        }
        else
        {
          *(idepthData+ii) = 0;
          *(idepthVarData+ii) = 0;
        }
      }

      std::string baseName = outputFolder + "/" + std::to_string(iterNum) 
                  + "_" + std::to_string(activeKeyFrame->timestamp()) 
                  + "_" + std::to_string(activeKeyFrame->id());

      std::ofstream depthFile(baseName + "_sparse_depth.bin", std::ios::binary | std::ios::out);
      std::ofstream depthVarFile(baseName + "_sparse_depthVar.bin", std::ios::binary | std::ios::out);

      double min, max;
      cv::minMaxLoc(*depthGt, &min, &max);

      cv::imwrite(baseName + "_rgb.png", *rgb);
      cv::imwrite(baseName + "_depthGT.png", *depthGt);
      depthFile.write(reinterpret_cast<const char*> (&idepthData), sizeof idepthData);
      depthVarFile.write(reinterpret_cast<const char*> (&idepthVarData), sizeof idepthVarData);

      //printf("%s _rgb.png\n", baseName.c_str());
      //ROS_WARN("cvImagesSet: %d", f->cvImagesSet());
      ROS_WARN("Iteration Number: %d", iterNum);
      printf("Keyframe Id: %d\n", activeKeyFrame->id());
      printf("Min depth gt: %f, Max depth gt: %f\n", min, max);
      //printf("Max idepth sparse scaled: %f\n", *std::max_element(idepthFloat, idepthFloat+size));
      printf("Max idepth sparse descaled: %f\n", maxDepth);
      //printf("Max idepthVar sparse scaled: %f\n", *std::max_element(idepthVarFloat, idepthVarFloat+size));
      printf("Max idepthVar sparse descaled: %f\n", maxDepthVar);
      //printf("scale obtained by ouput saver: %f\n", scale);

    }

    if(predictDepth)
    {

      /** Store the currentDepthMap propagated inside otherDepthMap **/
      memcpy(otherDepthMap,currentDepthMap,width*height*sizeof(DepthMapPixelHypothesis));

      if (depthCompletion)
      {
        /* Use the depth completion network to update depths */

        /** Create a float array of idepth and idepth variance **/
        std::vector<float> idepth(width*height, 0.0);
        std::vector<float> idepthVar(width*height, 0.0);
        //float idepthVar[width*height];
        fillIdepthArray(idepth.data(), idepthVar.data());

        /** Get the depth prediction from network **/
        //** Call Depth Node **//
        ROS_WARN("Predicting depth using depthCompletion network");
        reinforced_visual_slam::DepthFusion srv;
        srv.request.rgb_image = *(cv_bridge::CvImage( std_msgs::Header(),"bgr8",*(activeKeyFrame->rgbMat()) ).toImageMsg());
        srv.request.idepth = idepth;
        srv.request.idepth_var = idepthVar;
        srv.request.scale = prev_scale;
        if(depthClient.call(srv)){
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
            return;
          }
          cv::Mat predicted_idepth;

          // Network returns predicted idepth of 320x240. Resize it to 640x480
          cv::resize(cvImage->image, predicted_idepth, cv::Size(width, height), 0, 0, cv::INTER_NEAREST);
          
          //** Update currentDepthMap using the predicted depthmap (Todo) **//
          // Convert the scale of fused_depth back to original (m)
          cv::Mat fused_depth(cv::Size(width, height), CV_32FC1, float(0));
          if(readSparse)
            setFromIdepthMapSparse(reinterpret_cast<float*>(predicted_idepth.data), 
              reinterpret_cast<float*>(fused_depth.data), prev_scale);
          else
          {
            cv::Mat predicted_variance;
            if (depthPredictionVariance < 0)
            {
              //** convert srv.response.predicted_depth to cv mat **//
              cv_bridge::CvImagePtr cvVariance;
              try
              {
                cvVariance = cv_bridge::toCvCopy(srv.response.residual, "");
              }
              catch (cv_bridge::Exception& e)
              {
                ROS_ERROR("cv_bridge exception when receiving residual message : %s", e.what());
                return;
              }
              // Network returns predicted idepth of 320x240. Resize it to 640x480
              cv::resize(cvVariance->image, predicted_variance, cv::Size(width, height), 0, 0, cv::INTER_NEAREST);
            }
            setFromIdepthMap(reinterpret_cast<float*>(predicted_idepth.data), 
              reinterpret_cast<float*>(fused_depth.data), prev_scale, reinterpret_cast<float*>(predicted_variance.data));
          }
        
          /** Save fused depth it inside frame **/
          activeKeyFrame->setCVDepth(fused_depth);
        }
        else
          ROS_ERROR("No response from depth predictor node!!");
      }
      else
      {
        /** Get the depth prediction from single image network **/
        //** Call Depth Node **//
        ROS_WARN("Predicting depth using single image network");
        reinforced_visual_slam::PredictDepthmap srv;
        srv.request.rgb_image = *(cv_bridge::CvImage( std_msgs::Header(),"bgr8",*(activeKeyFrame->rgbMat()) ).toImageMsg());
        if(singleImageDepthClient.call(srv)){
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
            return;
          }
          cv::Mat predicted_idepth, predicted_variance;
          // If Network returns predicted idepth of 320x240, uncomment below to resize it to 640x480
          //cv::resize(cvImage->image, predicted_idepth, cv::Size(width, height));
          
          // Multiplying the predicted idepth by the previous depthmap's scale. LSD SLAM assumes idepth to be at this scale!
          predicted_idepth = cvImage->image * prev_scale;

          //** Get the predicted variance if set to be used **//
          if (depthPredictionVariance < 0)
          {
            //** convert srv.response.predicted_depth to cv mat **//
            cv_bridge::CvImagePtr cvVariance;
            try
            {
              cvVariance = cv_bridge::toCvCopy(srv.response.residual, "");
            }
            catch (cv_bridge::Exception& e)
            {
              ROS_ERROR("cv_bridge exception when receiving residual message : %s", e.what());
              return;
            }
            // changing scale to last keyframe's
            predicted_variance = cvImage->image * (prev_scale*prev_scale);
          }


          /** Fuse predicted and projected (*otherDepthMap) idepths to make combined depthmap **/
          // Mat to store combined (predicted + projected) depth
          cv::Mat combined_depth(cv::Size(width, height), CV_32FC1, float(0));
          fuseDepthMapsManual(reinterpret_cast<float*>(predicted_idepth.data), 
            reinterpret_cast<float*>(combined_depth.data), reinterpret_cast<float*>(predicted_variance.data));

          //** Sparsify combined depth and update currentDepthMap (Todo) **//
          //new_keyframe->setDepthFromGroundTruth(reinterpret_cast<float*>(combined_depth.data));

          /** Convert the scale of combined_depth back to original (m) and save it inside frame **/
          combined_depth *= prev_scale;
          activeKeyFrame->setCVDepth(combined_depth);
        }

        if (printDepthPredictionDebugs)
        {
          ROS_DEBUG("srv.response.depthmap.step: %d", srv.response.depthmap.step);
          ROS_DEBUG("srv.response.depthmap.Header: %d", srv.response.depthmap.header.seq);
          ROS_DEBUG("srv.response.depthmap.encoding: %s", srv.response.depthmap.encoding.c_str());
          /**
          //Normal Plotting for debugs:
          ROS_DEBUG("predicted_idepth encoding: %d",predicted_idepth.type());
          ROS_DEBUG("Converted size predicted_idepth rows:%d and cols:%d", predicted_idepth.rows, predicted_idepth.cols);

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
      }
    }
  }
  else
  {
    // Initialize from ground truth
    activeKeyFrame = new_keyframe;
    activeKeyFramelock = activeKeyFrame->getActiveLock();
    activeKeyFrameImageData = new_keyframe->image(0);
    activeKeyFrameIsReactivated = false;
    cv::Mat kfGtDepth = new_keyframe->depthGTMat()->clone()/prev_scale;
    activeKeyFrame->setDepthFromGroundTruth(reinterpret_cast<float*>(kfGtDepth.data));
    initializeFromGTDepth(activeKeyFrame);
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
  if (isnanf(rescaleFactor) || rescaleFactor < 0)
  {
    exitSystem = true;
    printf("Depth map rescaleFactor has become unacceptable. shutting down SLAM system\n");
    return;
  }
  activeKeyFrame->pose->thisToParent_raw = sim3FromSE3(oldToNew_SE3.inverse(), rescaleFactor);
  activeKeyFrame->pose->invalidateCache();

  if(predictDepth && plotDepthFusion){
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

    /** Converting the scale of depth gt for plotting **/
    cv::Mat depth_gt = activeKeyFrame->depthGTMat()->clone();
    depth_gt *= float(1/(rescaleFactor * prev_scale));
    debugPlotsDepthFusion(reinterpret_cast<float*>(depth_gt.data));
    //Util::displayImage( "iDEPTH Propagated", debugIdepthPropagated, true );
    Util::displayImage( "iDEPTH Combined", debugIdepthFused, true );
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

void DepthMapPredictor::fuseDepthMapsManual(const float* idepth_predicted, float* depth_combined, 
  const float* idepth_variance){
  /** 
  otherDepthMap is the projected idepthmap, idepth_predicted is the predicted idepthmap. 
  This function updates the idepth and variance of currentDepthMap based on otherDepthMap.
  Also, it computes a new float* depth_combined.
  **/
  bool use_predicted_variance = (depthPredictionVariance < 0);
  float idepth_var_predicted;

  for(int y=0;y<height;y++)
  {
    for(int x=0;x<width;x++)
    {
      // Check if projected values exists for this pixel. If so, merge with predicted values
      if(otherDepthMap[x+y*width].isValid  && otherDepthMap[x+y*width].idepth > 0 && !isnanf(otherDepthMap[x+y*width].idepth))
      {
        // Check if the value predicted for this pixel is valid. If so, merge with projected values
        if(!isnanf(idepth_predicted[x+y*width]) && (idepth_predicted[x+y*width] > 0))
        {
          if(use_predicted_variance)
            idepth_var_predicted = idepth_variance[x+y*width];
          else
            idepth_var_predicted = depthPredictionVariance;
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
        if(!isnanf(idepth_predicted[x+y*width]) && (idepth_predicted[x+y*width] > 0))
        {
          if(use_predicted_variance)
            idepth_var_predicted = idepth_variance[x+y*width];
          else
            idepth_var_predicted = depthPredictionVariance;
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

void DepthMapPredictor::debugPlotsDepthFusion(const float* depth_gt)
{
  if(activeKeyFrame == 0) return;

  cv::Mat keyFrameImage(activeKeyFrame->height(), activeKeyFrame->width(), CV_32F, const_cast<float*>(activeKeyFrameImageData));

  keyFrameImage.convertTo(debugIdepthPropagated, CV_8UC1);
  cv::cvtColor(debugIdepthPropagated, debugIdepthPropagated, CV_GRAY2RGB);
  keyFrameImage.convertTo(debugIdepthFused, CV_8UC1);
  cv::cvtColor(debugIdepthFused, debugIdepthFused, CV_GRAY2RGB);
  keyFrameImage.convertTo(debugIdepthGt, CV_8UC1);
  cv::cvtColor(debugIdepthGt, debugIdepthGt, CV_GRAY2RGB);

  // debug plot & publish sparse version?
  int refID = referenceFrameByID_offset;


  for(int y=0;y<height;y++)
  {
    for(int x=0;x<width;x++)
    {
      int idx = x + y*width;

      if(currentDepthMap[idx].isValid) {
        cv::Vec3b color = currentDepthMap[idx].getVisualizationColor(refID);
        debugIdepthFused.at<cv::Vec3b>(y,x) = color;
      }

      if(otherDepthMap[idx].isValid) {
        cv::Vec3b color = otherDepthMap[idx].getVisualizationColor(refID);
        debugIdepthPropagated.at<cv::Vec3b>(y,x) = color;
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

void DepthMapPredictor::fillIdepthArray(float* idepth, float* idepthVar)
{
  for(int y=0;y<height;y++)
  {
    for(int x=0;x<width;x++)
    { 
      // Check if projected values exists for this pixel. If so, write to float array
      if(currentDepthMap[x+y*width].isValid  && currentDepthMap[x+y*width].idepth_smoothed > 0 && 
        !isnanf(currentDepthMap[x+y*width].idepth_smoothed))
      {
        idepth[x+y*width] = currentDepthMap[x+y*width].idepth_smoothed;
        idepthVar[x+y*width] = currentDepthMap[x+y*width].idepth_var_smoothed;
      }
      else
      {
        idepth[x+y*width] = 0;
        idepthVar[x+y*width] = 0;
      }
    }

  }
}

void DepthMapPredictor::setFromIdepthMapSparse(const float* idepth_predicted, float* depth_fused, float scale)
{
  /** 
  idepth_predicted is the fused idepthmap. 
  This function updates the idepth and variance of currentDepthMap based on otherDepthMap (at current scale).
  Also, it computes a new float* depth_fused (in m).
  **/
  for(int y=0;y<height;y++)
  {
    for(int x=0;x<width;x++)
    {      
      // Check if the value predicted for this pixel is valid. If so, merge with projected values
      if(!isnanf(idepth_predicted[x+y*width]) && idepth_predicted[x+y*width] > 0)
      {
        //Check if the value exists at the point for current depthmap, if yes replace it with prediction!
        if(currentDepthMap[x+y*width].isValid  
          && currentDepthMap[x+y*width].idepth > 0 && !isnanf(currentDepthMap[x+y*width].idepth))
        {
          float idepth_scaled = idepth_predicted[x+y*width] * scale;
          currentDepthMap[x+y*width].idepth = idepth_scaled;
          currentDepthMap[x+y*width].idepth_smoothed = idepth_scaled;
        }
        depth_fused[x+y*width] = 1.0f/idepth_predicted[x+y*width];
      }
      // Else, make current value zero for the dense depth map
      else
      {
        depth_fused[x+y*width] = 0;
      }
    }
  }
}

void DepthMapPredictor::setFromIdepthMap(const float* idepth_predicted, float* depth_fused, 
  float scale, const float* idepth_variance)
{
  /** 
  idepth_predicted is the fused idepthmap. 
  This function updates the idepth and variance of currentDepthMap based on otherDepthMap (at current scale).
  Also, it computes a new float* depth_fused (in m).
  **/
  bool use_predicted_variance = (depthPredictionVariance < 0);
  float idepth_var_predicted;

  for(int y=0;y<height;y++)
  {
    for(int x=0;x<width;x++)
    {      
      // Check if the value predicted for this pixel is valid. If so, merge with projected values
      if(!isnanf(idepth_predicted[x+y*width]) && (idepth_predicted[x+y*width] > 0))
      {
        if(use_predicted_variance)
          idepth_var_predicted = idepth_variance[x+y*width] * (scale*scale);
        else
          idepth_var_predicted = depthPredictionVariance * (scale*scale);
        float idepth_scaled = idepth_predicted[x+y*width] * scale;

        currentDepthMap[x+y*width] = DepthMapPixelHypothesis(
            idepth_scaled,
            idepth_scaled,
            idepth_var_predicted,
            idepth_var_predicted,
            20);

        depth_fused[x+y*width] = 1.0f/idepth_predicted[x+y*width];
      }
      // Else, make current value invalid
      else
      {
        currentDepthMap[x+y*width].isValid = false;
        currentDepthMap[x+y*width].blacklisted = 0;
        depth_fused[x+y*width] = 0;
      }
    }
  }
}