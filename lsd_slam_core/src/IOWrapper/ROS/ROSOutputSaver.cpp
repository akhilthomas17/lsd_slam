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

	std::string baseName = resultFolder + "/" + std::to_string(itrNum) 
							+ "_" + std::to_string(f->timestamp()) 
							+ "_" + std::to_string(f->id());

	std::ofstream depthFile(baseName + "_sparse_depth.bin", std::ios::binary | std::ios::out);
	std::ofstream depthVarFile(baseName + "_sparse_depthVar.bin", std::ios::binary | std::ios::out);

	double min, max;
	cv::minMaxLoc(*depthGt, &min, &max);

	cv::imwrite(baseName + "_rgb.png", *rgb);
	cv::imwrite(baseName + "_depthGT.png", *depthGt);
	depthFile.write(reinterpret_cast<const char*> (&idepthData), sizeof idepthData);
	depthVarFile.write(reinterpret_cast<const char*> (&idepthVarData), sizeof idepthVarData);

	if (debug)
	{
		//printf("%s _rgb.png\n", baseName.c_str());
		//ROS_WARN("cvImagesSet: %d", f->cvImagesSet());
		printf("Iteration Number: %d\n", itrNum);
		printf("Keyframe Id: %d\n", f->id());
		ROS_WARN("Min depth gt: %f, Max depth gt: %f", min, max);
		//printf("Max idepth sparse scaled: %f\n", *std::max_element(idepthFloat, idepthFloat+size));
		printf("Max idepth sparse descaled: %f\n", maxDepth);
		//printf("Max idepthVar sparse scaled: %f\n", *std::max_element(idepthVarFloat, idepthVarFloat+size));
		printf("Max idepthVar sparse descaled: %f\n", maxDepthVar);
		//printf("scale obtained by ouput saver: %f\n", scale);
	}

}

}