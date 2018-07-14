/**
*/

#include "LiveSLAMWrapper.h"

#include <boost/thread.hpp>
#include "util/settings.h"
#include "util/globalFuncs.h"
#include "SlamSystem.h"
#include "SlamSystemReinforced.h"
#include "DataStructures/Frame.h"

#include <sstream>
#include <fstream>
#include <dirent.h>
#include <algorithm>
#include <sys/stat.h>


#include "IOWrapper/ROS/ROSOutputSaver.h"
#include "IOWrapper/ROS/ROSOutput3DWrapper.h"
#include "IOWrapper/ROS/rosReconfigure.h"
#include "IOWrapper/OpenCV/ImageDisplay_OpenCV.cpp"

#include "util/Undistorter.h"
#include <ros/package.h>


int getComboFileRgbdTUM (std::string source, std::vector<std::string> &rgb_files, 
	std::vector<std::string> &depth_files, std::vector<double> &timestamps, std::string basename, bool basename_received)
{
    std::vector < std::vector<std::string>* > files;
    files.push_back(&rgb_files);
    files.push_back(&depth_files);

    std::ifstream f(source.c_str());

    if (f.good() && f.is_open()) {
        while (!f.eof()) {
            std::string l;

            std::getline(f, l);


            if (l == "" || l[0] == '#')
                continue;

            // Split the string to separate timestamps and rgb image file name
            std::istringstream iss(l);
            std::vector<std::string> tokens{std::istream_iterator<std::string>{iss},
                                            std::istream_iterator<std::string>{}};
            timestamps.push_back(std::atof(tokens[0].c_str()));
            files[0]->push_back(tokens[1]);
            files[1]->push_back(tokens[3]);
        }

        f.close();
        std::string prefix;

        if(basename_received)
        {
        	prefix = basename;
        }
        else
        {
        	size_t sp = source.find_last_of('/');
	        if (sp == std::string::npos)
	            prefix = "";
	        else
	            prefix = source.substr(0, sp);
        }

        for (int ii = 0; ii < 2; ii++)
        {
	        for (unsigned int i = 0; i < files[ii]->size(); i++) {
	            if (files[ii]->at(i)[0] != '/')
	                files[ii]->at(i) = prefix + "/" + files[ii]->at(i);
	        }
	    }

    } else {
        f.close();
        return -1;
    }
    std::cout << "Depth images: " << depth_files.size() << std::endl;
    std::cout << "RGB images: " << rgb_files.size() << std::endl;
    return (int) rgb_files.size();
}

using namespace lsd_slam;


int main( int argc, char** argv )
{
	bool _debug = false;
	ros::init(argc, argv, "LSD_SLAM_TRAIN_DATA_GEN");

	dynamic_reconfigure::Server<lsd_slam_core::LSDParamsConfig> srv(ros::NodeHandle("~"));
	srv.setCallback(dynConfCb);

	dynamic_reconfigure::Server<lsd_slam_core::LSDDebugParamsConfig> srvDebug(ros::NodeHandle("~Debug"));
	srvDebug.setCallback(dynConfCbDebug);

	packagePath = ros::package::getPath("lsd_slam_core")+"/";

	// get camera calibration in form of an undistorter object.
	// if no undistortion is required, the undistorter will just pass images through.
	std::string calibFile;
	Undistorter* undistorter = 0;
	if(ros::param::get("~calib", calibFile))
	{
		 undistorter = Undistorter::getUndistorterForFile(calibFile.c_str());
		 ros::param::del("~calib");
	}

	if(undistorter == 0)
	{
		printf("need camera calibration file! (set using _calib:=FILE)\n");
		exit(0);
	}

	int w = undistorter->getOutputWidth();
	int h = undistorter->getOutputHeight();

	int w_inp = undistorter->getInputWidth();
	int h_inp = undistorter->getInputHeight();

	float fx = undistorter->getK().at<double>(0, 0);
	float fy = undistorter->getK().at<double>(1, 1);
	float cx = undistorter->getK().at<double>(2, 0);
	float cy = undistorter->getK().at<double>(2, 1);
	Sophus::Matrix3f K;
	K << fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0;

	int itr_num;
	if(!ros::param::get("~itr_num", itr_num))
		itr_num = 0;

	// open dataset sequence file (Assumed to be in RGBD sequence format)
	std::string source, basename;
	bool basename_received = false;
	std::vector<std::string> rgb_files, depth_files;
	// For reading timestamps from file
	std::vector<double> timestamps;

	if(!ros::param::get("~files", source))
	{
		printf("need source rgb_files! (set using _files:=FOLDER)\n");
		exit(0);
	}
	ros::param::del("~files");

	if(ros::param::get("~basename", basename))
	{
		basename_received = true;
		printf("Basename provided. Taking downloaded files from %s \n", basename);
	}
	ros::param::del("~basename");

	// make output folder with the dataset name
	size_t pos = source.find_last_of('/');
	std::string datsetName = source.substr(0, pos);
	pos = datsetName.find_last_of('/');
	datsetName = datsetName.substr(pos+1);

	std::string resultFolder = "/misc/lmbraid19/thomasa/datasets/LSDDepthTraining/" + datsetName;

	// check program mode selectors
	doSlam = true;
	freeDebugParam1 = 0.0001;
	displayDepthMap = false;
	predictDepth = false;
	writeDepthToFile = true;
	ROS_WARN("doSLAM_mode: %d", doSlam);
	ROS_WARN("minUseGrad: %f", minUseGrad);
	ROS_WARN("freeDebugParam1 (Variance of depth prediction): %f", VAR_GT_INIT_INITIAL);
	ROS_WARN("displayDepthMap: %d", displayDepthMap);
	ROS_WARN("predictDepth: %d", predictDepth);
	ROS_WARN("writeDepthToFile: %d", writeDepthToFile);

	/** To include the old way!!
	Output3DWrapper* outputWrapper;
	SlamSystem* system;

	if(!writeDepthToFile)
	{
		ROS_WARN("Creating a SlamSystem node..!");
		printf("writeDepthToFile: %d\n",writeDepthToFile);
		// make output wrapper. just set to zero if no output is required.
		outputWrapper = new ROSOutputSaver(w, h, itr_num, resultFolder);
		// make slam system
		system = new SlamSystem(w, h, K, doSlam);
		system->init(w, h, K);
	}
	else
	{
	**/
	ROS_WARN("Creating a SlamSystemReinforced node..!");
	Output3DWrapper* outputWrapper = new ROSOutput3DWrapper(w, h);
	outputFolder = resultFolder;
	iterNum = itr_num;
	SlamSystemReinforced* system = new SlamSystemReinforced(w, h, K, doSlam);
	//}
	system->setVisualization(outputWrapper);

	// creating the result folder
	int status = mkdir(resultFolder.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
	if (status==0)
		ROS_INFO("Created directory %s", resultFolder.c_str());
	else if (status==EEXIST)
		ROS_INFO("Directory %s exists", resultFolder.c_str());
	else
		ROS_INFO("Error %d in directory creation", status);


	if(getComboFileRgbdTUM(source, rgb_files, depth_files, timestamps, basename, basename_received) >= 0)
	{
		printf("found %d rgb and depth files from file %s!\n", (int)rgb_files.size(), source.c_str());
	}
	else
	{
		printf("could not load file list! wrong path / file?\n");
	}

	// get HZ
	double hz = 0;
	if(!ros::param::get("~hz", hz))
		hz = 0;
	ros::param::del("~hz");

	int start_indx;
	if(!ros::param::get("~start_indx", start_indx))
		start_indx = 0;
	ros::param::del("~start_indx");

	int len_traj;
	if(!ros::param::get("~len_traj", len_traj))
		len_traj = rgb_files.size();
	ros::param::del("~len_traj");

	cv::Mat image = cv::Mat(h,w,CV_8U);
	int runningIDX=0;
	double fakeTimeStamp = 0;

	ros::Rate r(hz);

	for(unsigned int i=start_indx;i<(start_indx+len_traj);i++)
	{
		cv::Mat imageRgb = cv::imread(rgb_files[i], CV_LOAD_IMAGE_UNCHANGED);
		cv::Mat imageDist;
		cvtColor(imageRgb, imageDist, CV_RGB2GRAY);
		cv::Mat depthImg = cv::imread(depth_files[i], CV_LOAD_IMAGE_UNCHANGED);

		if(imageDist.rows != h_inp || imageDist.cols != w_inp)
		{
			if(imageDist.rows * imageDist.cols == 0)
				printf("failed to load image %s! skipping.\n", rgb_files[i].c_str());
			else
				printf("image %s has wrong dimensions - expecting %d x %d, found %d x %d. Skipping.\n",
						rgb_files[i].c_str(),
						w,h,imageDist.cols, imageDist.rows);
			continue;
		}
		assert(imageDist.type() == CV_8U);

		undistorter->undistort(imageDist, image);
		assert(image.type() == CV_8U);

        fakeTimeStamp = timestamps[i];

		if(runningIDX == 0)
		{
			cv::Mat init_image;
            depthImg.convertTo(init_image, CV_32F, 0.0002);
            system->gtDepthInit(image.data, reinterpret_cast<float*>(init_image.data), fakeTimeStamp, runningIDX);
        } 
        else
        {
			system->trackFrameLSD(&imageRgb, &depthImg, runningIDX, hz == 0, fakeTimeStamp);
        }

		if(_debug)
			ROS_WARN("runningIDX: %d", runningIDX);

		runningIDX++;

		if(hz != 0)
			r.sleep();

		ros::spinOnce();

		if(!ros::ok())
			break;
	}


	delete system;
	delete undistorter;
	delete outputWrapper;
	return 0;
}
