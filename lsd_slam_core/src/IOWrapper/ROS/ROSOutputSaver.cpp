#include "ROSOutputSaver.h"
#include <half.hpp>

#include "DataStructures/Frame.h"

#include <fstream>
#include <sys/stat.h>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>

using half_float::half;

namespace lsd_slam
{

ROSOutputSaver::ROSOutputSaver(int width, int height, int newItrNum, std::string newResultFolder): ROSOutput3DWrapper(width, height)
{
	itrNum = newItrNum;
	debug = true;
	
	/** creating result folder **/
	resultFolder = newResultFolder;
	int status = mkdir(resultFolder.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
	if (status==0)
		ROS_INFO("Created directory %s", resultFolder.c_str());
	else if (status==EEXIST)
		ROS_INFO("Directory %s exists", resultFolder.c_str());
	else
		ROS_INFO("Error %d in directory creation", status);
}

ROSOutputSaver::~ROSOutputSaver()
{

}

void ROSOutputSaver::publishKeyframe(Frame* f)
{
	float scale = f->getScaledCamToWorld().scale();
	cv::Mat* depthGt = f->depthMat();
	//depthGt->convertTo(*depthGt, CV_16U, 5000); This was not needed inside pure LSD SLAM
	cv::Mat* rgb = f->rgbMat();

	// De-scale the idepth and idepthVar stored back to world (m)
	const float* idepthFloat = f->idepth();
	const float* idepthVarFloat = f->idepthVar();
	int size = width*height;

	half depthData[size];
	half depthVarData[size];

	float maxDepth = 0;
	float maxDepthVar = 0;

	for (int ii=0; ii < size; ii++)
	{
		if (*(idepthFloat + ii) > 0 && *(idepthVarFloat + ii)>0)
		{
			*(depthData+ii) = 1 / (*(idepthFloat+ii)/scale);
			*(depthVarData+ii) = 1 / (*(idepthVarFloat+ii)/scale);
			if(*(depthData+ii) > maxDepth)
				maxDepth = *(depthData+ii);
			if(*(depthVarData+ii) > maxDepthVar)
				maxDepthVar = *(depthVarData+ii);
		}
		else
		{
			*(depthData+ii) = 0;
			*(depthVarData+ii) = 0;
		}
	}

	std::string baseName = resultFolder + "/" + std::to_string(itrNum) 
							+ "_" + std::to_string(f->timestamp()) 
							+ "_" + std::to_string(f->id());

	std::ofstream depthFile(baseName + "_sparse_depth.bin", std::ios::binary | std::ios::out);
	std::ofstream depthVarFile(baseName + "_sparse_depthVar.bin", std::ios::binary | std::ios::out);

	double min, max;
	cv::minMaxLoc(*depthGt, &min, &max);

	cv::imwrite(baseName + "_rgb.png", *rgb);
	cv::imwrite(baseName + "_depthGT.png", *depthGt);
	depthFile.write(reinterpret_cast<const char*> (&depthData), sizeof depthData);
	depthVarFile.write(reinterpret_cast<const char*> (&depthVarData), sizeof depthVarData);

	if (debug)
	{
		//printf("%s _rgb.png\n", baseName.c_str());
		//ROS_WARN("cvImagesSet: %d", f->cvImagesSet());
		printf("Keyframe Id: %d\n", f->id());
		ROS_WARN("Min depth gt: %f, Max depth gt: %f", min, max);
		printf("Max depth sparse: %f\n", maxDepth);
		printf("Max depthVar sparse: %f\n", maxDepthVar);
		printf("scale obtained by ouput saver: %f\n", scale);
	}

}

}