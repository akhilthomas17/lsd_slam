#include "ROSOutputSaver.h"
#include <half.hpp>

#include "DataStructures/Frame.h"

#include "std_msgs/Float32MultiArray.h"
#include "lsd_slam_viewer/keyframeGraphMsg.h"
#include "lsd_slam_viewer/keyframeMsg.h"

// for siasa visualization
#include <reinforced_visual_slam/keyframeMsgSiasa.h>
#include <cv_bridge/cv_bridge.h>


#include <fstream>
#include <sys/stat.h>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include "util/settings.h"

using half_float::half;

namespace lsd_slam
{

ROSOutputSaver::ROSOutputSaver(int width, int height, int newItrNum, std::string newResultFolder):ROSOutput3DWrapper(width, height)
{
	itrNum = newItrNum;
	debug = true;
	
	/** creating result folder **/
	resultFolder = newResultFolder;
}

ROSOutputSaver::ROSOutputSaver(int width, int height):ROSOutput3DWrapper(width, height)
{
	itrNum = 0;
	debug = true;
	
	/** creating result folder **/
	resultFolder = outputFolder;
}

ROSOutputSaver::~ROSOutputSaver()
{

}

void ROSOutputSaver::publishKeyframe(Frame* f)
{
	if (writeTestDepths && (f->id() > 0)){

		float scale = f->getScaledCamToWorld().scale();
		cv::Mat depthGt = f->depthGTMat()->clone();
		cv::Mat* depthPrediction;
		cv::Mat* rgb;
		if (testMode)
		{
			if (predictDepth){
				depthPrediction = f->depthMat();
				depthPrediction->convertTo(*depthPrediction, CV_16U, 5000);
			}
			depthGt.convertTo(depthGt, CV_16U, 5000); // This was not needed inside pure LSD SLAM
		}
		else
			rgb = f->rgbMat();

		// De-scale the idepth and idepthVar stored back to world (m)
		const float* idepthFloat = f->idepth();
		const float* idepthVarFloat = f->idepthVar();
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

		std::string baseName;

		if(testMode)
			baseName = resultFolder + "/" + std::to_string(f->timestamp()) 
								+ "_" + std::to_string(f->id());
		else
			baseName = resultFolder + "/" + std::to_string(itrNum) 
								+ "_" + std::to_string(f->timestamp()) 
								+ "_" + std::to_string(f->id());

		ROS_WARN("baseName: %s\n", baseName.c_str());

		std::ofstream depthFile(baseName + "_sparse_depth.bin", std::ios::binary | std::ios::out);
		std::ofstream depthVarFile(baseName + "_sparse_depthVar.bin", std::ios::binary | std::ios::out);

		double min, max;
		cv::minMaxLoc(depthGt, &min, &max);

		if(testMode && predictDepth)
			cv::imwrite(baseName + "_depthPrediction.png", *depthPrediction);
		else if (!testMode)
			cv::imwrite(baseName + "_rgb.png", *rgb);
		cv::imwrite(baseName + "_depthGT.png", depthGt);
		depthFile.write(reinterpret_cast<const char*> (&idepthData), sizeof idepthData);
		depthVarFile.write(reinterpret_cast<const char*> (&idepthVarData), sizeof idepthVarData);

		if (debug)
		{
			//printf("%s _rgb.png\n", baseName.c_str());
			//ROS_WARN("cvImagesSet: %d", f->cvImagesSet());
			if (!testMode){
				printf("Iteration Number: %d\n", itrNum);
				printf("Max idepthVar sparse descaled: %f\n", maxDepthVar);
			}
			printf("Keyframe Id: %d\n", f->id());
			ROS_WARN("Min depth gt: %f, Max depth gt: %f", min, max);
			//printf("Max idepth sparse scaled: %f\n", *std::max_element(idepthFloat, idepthFloat+size));
			printf("Max idepth sparse descaled: %f\n", maxDepth);
			//printf("Max idepthVar sparse scaled: %f\n", *std::max_element(idepthVarFloat, idepthVarFloat+size));
			//printf("scale obtained by ouput saver: %f\n", scale);
		}
	}
	else
	{
		lsd_slam_viewer::keyframeMsg fMsg;


		boost::shared_lock<boost::shared_mutex> lock = f->getActiveLock();

		fMsg.id = f->id();
		fMsg.time = f->timestamp();
		fMsg.isKeyframe = true;

		int w = f->width(publishLvl);
		int h = f->height(publishLvl);

		memcpy(fMsg.camToWorld.data(),f->getScaledCamToWorld().cast<float>().data(),sizeof(float)*7);
		fMsg.fx = f->fx(publishLvl);
		fMsg.fy = f->fy(publishLvl);
		fMsg.cx = f->cx(publishLvl);
		fMsg.cy = f->cy(publishLvl);
		fMsg.width = w;
		fMsg.height = h;


		fMsg.pointcloud.resize(w*h*sizeof(InputPointDense));

		InputPointDense* pc = (InputPointDense*)fMsg.pointcloud.data();

		const float* idepth = f->idepth(publishLvl);
		const float* idepthVar = f->idepthVar(publishLvl);
		const float* color = f->image(publishLvl);

		for(int idx=0;idx < w*h; idx++)
		{
			pc[idx].idepth = idepth[idx];
			pc[idx].idepth_var = idepthVar[idx];
			pc[idx].color[0] = color[idx];
			pc[idx].color[1] = color[idx];
			pc[idx].color[2] = color[idx];
			pc[idx].color[3] = color[idx];
		}

		keyframe_publisher.publish(fMsg);

		reinforced_visual_slam::keyframeMsgSiasa kfMsg;
		
		kfMsg.id = f->id();
		kfMsg.time = f->timestamp();
		
		memcpy(kfMsg.camToWorld.data(),f->getScaledCamToWorld().cast<float>().data(),sizeof(float)*7);
		
		kfMsg.fx = f->fx(publishLvl);
		kfMsg.fy = f->fy(publishLvl);
		kfMsg.cx = f->cx(publishLvl);
		kfMsg.cy = f->cy(publishLvl);
		kfMsg.width = w;
		kfMsg.height = h;

		kfMsg.rgb = *(cv_bridge::CvImage( std_msgs::Header(),"bgr8",*(f->rgbMat()) ).toImageMsg());
		kfMsg.depth = *(cv_bridge::CvImage( std_msgs::Header(),"32FC1",*(f->depthMat()) ).toImageMsg());
		siasa_publisher.publish(kfMsg);
	}

}

}