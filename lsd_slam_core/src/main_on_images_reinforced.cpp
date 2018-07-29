/**
* This file is part of LSD-SLAM.
*
* Copyright 2013 Jakob Engel <engelj at in dot tum dot de> (Technical University of Munich)
* For more information see <http://vision.in.tum.de/lsdslam> 
*
* LSD-SLAM is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* LSD-SLAM is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with LSD-SLAM. If not, see <http://www.gnu.org/licenses/>.
*/

#include "LiveSLAMWrapperReinforced.h"

#include <boost/thread.hpp>
#include "util/settings.h"
#include "util/globalFuncs.h"
#include "SlamSystemReinforced.h"

#include <sstream>
#include <fstream>
#include <dirent.h>
#include <algorithm>

#include "IOWrapper/ROS/ROSOutput3DWrapper.h"
#include "IOWrapper/ROS/ROSOutputSaver.h"
#include "IOWrapper/ROS/rosReconfigure.h"

#include "util/Undistorter.h"
#include <ros/package.h>
#include <ros/console.h>

#include "opencv2/opencv.hpp"

std::string &ltrim(std::string &s) {
        s.erase(s.begin(), std::find_if(s.begin(), s.end(), std::not1(std::ptr_fun<int, int>(std::isspace))));
        return s;
}
std::string &rtrim(std::string &s) {
        s.erase(std::find_if(s.rbegin(), s.rend(), std::not1(std::ptr_fun<int, int>(std::isspace))).base(), s.end());
        return s;
}
std::string &trim(std::string &s) {
        return ltrim(rtrim(s));
}
int getdir (std::string dir, std::vector<std::string> &files)
{
    DIR *dp;
    struct dirent *dirp;
    if((dp  = opendir(dir.c_str())) == NULL)
    {
        return -1;
    }

    while ((dirp = readdir(dp)) != NULL) {
    	std::string name = std::string(dirp->d_name);

    	if(name != "." && name != "..")
    		files.push_back(name);
    }
    closedir(dp);


    std::sort(files.begin(), files.end());

    if(dir.at( dir.length() - 1 ) != '/') dir = dir+"/";
	for(unsigned int i=0;i<files.size();i++)
	{
		if(files[i].at(0) != '/')
			files[i] = dir + files[i];
	}

    return files.size();
}

int getFile (std::string source, std::vector<std::string> &files)
{
	std::ifstream f(source.c_str());

	if(f.good() && f.is_open())
	{
		while(!f.eof())
		{
			std::string l;
			std::getline(f,l);

			l = trim(l);

			if(l == "" || l[0] == '#')
				continue;

			files.push_back(l);
		}

		f.close();

		size_t sp = source.find_last_of('/');
		std::string prefix;
		if(sp == std::string::npos)
			prefix = "";
		else
			prefix = source.substr(0,sp);

		for(unsigned int i=0;i<files.size();i++)
		{
			if(files[i].at(0) != '/')
				files[i] = prefix + "/" + files[i];
		}

		return (int)files.size();
	}
	else
	{
		f.close();
		return -1;
	}

}

int getFileRgbdTUM (std::string rgb_path, std::vector<std::string> &rgb_files, std::vector<std::string> &depth_files, std::vector<double> &timestamps)
{
    std::vector <std::string> sources;
    sources.push_back(rgb_path);
	sources.push_back(rgb_path.erase(rgb_path.find("rgb.txt")) + "depth.txt");

    std::vector < std::vector<std::string>* > files;
    files.push_back(&rgb_files);
    files.push_back(&depth_files);

    for (int ii = 0; ii < 2; ++ii) {
        std::string source = sources[ii];
        std::ifstream f(source.c_str());

        if (f.good() && f.is_open()) {
            while (!f.eof()) {
                std::string l;

                std::getline(f, l);

                //l = trim(l);

                if (l == "" || l[0] == '#')
                    continue;

                // Split the string to separate timestamps and rgb image file name
                std::istringstream iss(l);
                std::vector<std::string> tokens{std::istream_iterator<std::string>{iss},
                                                std::istream_iterator<std::string>{}};
                if (ii==0)
                    timestamps.push_back(std::atof(tokens[0].c_str()));
                files[ii]->push_back(tokens[1]);
            }

            f.close();

            size_t sp = source.find_last_of('/');
            std::string prefix;
            if (sp == std::string::npos)
                prefix = "";
            else
                prefix = source.substr(0, sp);

            for (unsigned int i = 0; i < files[ii]->size(); i++) {
                if (files[ii]->at(i)[0] != '/')
                    files[ii]->at(i) = prefix + "/" + files[ii]->at(i);
            }

        } else {
            f.close();
            return -1;
        }
    }
    std::cout << "Depth images: " << depth_files.size() << std::endl;
    std::cout << "RGB images: " << rgb_files.size() << std::endl;
    return (int) rgb_files.size();
}

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
	ros::init(argc, argv, "LSD_SLAM");

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

	// open dataset sequence file (Assumed to be in RGBD sequence format)
	std::string source, basename;
	bool basename_received = false;
	std::vector<std::string> rgb_files, depth_files;
	// For reading timestamps from file
	std::vector<double> timestamps;

	bool waitOnStart = false;

	if(ros::param::get("~waitOnStart", waitOnStart))
	{
		printf("Wait on start: %d \n", waitOnStart);
	}
	ros::param::del("~waitOnStart");

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

	if(ros::param::get("~writeTestDepths", writeTestDepths))
	{
		printf("writeTestDepths: %d \n", writeTestDepths);
	}
	ros::param::del("~writeTestDepths");

	if(ros::param::get("~outputFolder", outputFolder))
		printf("Received outputFolder\n");
	ros::param::del("~outputFolder");

	if( ros::param::get("~showDebugWindow", displayDepthMap))
		printf("Received debug depth map\n");
	ros::param::del("~showDepthWindow");

	if (!ros::param::get("~depthPredictionVariance", depthPredictionVariance))
		depthPredictionVariance = -1.0f;
	ros::param::del("~depthPredictionVariance");

	// make output wrapper. just set to zero if no output is required.

	Output3DWrapper* outputWrapper;

	if (testMode)
		outputWrapper = new ROSOutputSaver(w,h);
	else
		outputWrapper = new ROSOutput3DWrapper(w,h);


	// make slam system
	SlamSystemReinforced* system = new SlamSystemReinforced(w, h, K, doSlam);
	system->setVisualization(outputWrapper);

	
	if(getComboFileRgbdTUM(source, rgb_files, depth_files, timestamps, basename, basename_received) >= 0)
	{
		printf("found %d rgb and depth files from file %s!\n", (int)rgb_files.size(), source.c_str());
	}
	else if(getFileRgbdTUM(source, rgb_files, depth_files, timestamps) >= 0)
	{
		printf("found %d image rgb_files in file %s!\n", (int)rgb_files.size(), source.c_str());
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

    // check program mode selectors
	ROS_WARN("predictDepth_mode: %d", predictDepth);
	ROS_WARN("doSLAM_mode: %d", doSlam);
	ROS_WARN("useGtDepth: %d", useGtDepth);
	ROS_WARN("gtBootstrap: %d", gtBootstrap);
	ROS_WARN("plotDepthFusion: %d", plotDepthFusion);
	ROS_WARN("minUseGrad: %f", minUseGrad);
	ROS_WARN("depthPredictionVariance: %f", depthPredictionVariance);
	ROS_WARN("optimizeDeepTAM: %d", optimizeDeepTAM);
	ROS_WARN("readSparse: %d", readSparse);
	ROS_WARN("depthCompletion: %d", depthCompletion);
	ROS_WARN("testMode: %d", testMode);
	ROS_WARN("outputFolder: %s", outputFolder.c_str());
	

	cv::Mat image = cv::Mat(h,w,CV_8U);
	int runningIDX=0;
	double fakeTimeStamp = 0;

	ros::Rate r(hz);

	for(unsigned int i=0;i<rgb_files.size();i++)
	{
		cv::Mat imageDist = cv::imread(rgb_files[i], CV_LOAD_IMAGE_UNCHANGED);

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
		//assert(imageDist.type() == CV_8U);

		undistorter->undistort(imageDist, image);
		//assert(image.type() == CV_8U);

        fakeTimeStamp = timestamps[i];
        //printf("%f \n", fakeTimeStamp);
        cv::Mat depthImg = cv::imread(depth_files[i], CV_LOAD_IMAGE_UNCHANGED);
        printf("Frame: %d\n", runningIDX);
        depthImg.convertTo(depthImg, CV_32F, 0.0002);

		if(runningIDX == 0){
            if(waitOnStart)
            {
            	printf("Press ENTER to continue...\n");
            	std::cin.ignore( std::numeric_limits<std::streamsize>::max(), '\n' );
            }
            if(gtBootstrap || useGtDepth)
            	system->gtDepthInit(image, depthImg, fakeTimeStamp, runningIDX);
            else if (predictDepth){
            	if(!system->initFromPrediction(image, depthImg, fakeTimeStamp, runningIDX))
            	{
            		ROS_ERROR("Cannot init from single image depth predictor. Please check if the script is running. Terminating the program!!");
            		break;
            	}
            }
            else
            	system->randomInit(image.data, fakeTimeStamp, runningIDX);
        } 
        else
        {
        	if(useGtDepth || predictDepth)
        		system->trackFrame(image, depthImg, runningIDX, hz == 0, fakeTimeStamp);
        	else
        		system->trackFrameLSD(image, depthImg, runningIDX, hz == 0, fakeTimeStamp);
        }
        if(exitSystem){
        	ROS_ERROR("System encountered failure, Terminating the program!!");
        	break;
        }
		runningIDX++;

		if(hz != 0)
			r.sleep();

		if(fullResetRequested)
		{

			printf("FULL RESET!\n");
			delete system;

			system = new SlamSystemReinforced(w, h, K, doSlam);
			system->init(w, h, K);
			system->setVisualization(outputWrapper);

			fullResetRequested = false;
			runningIDX = 0;
		}

		ros::spinOnce();

		if(!ros::ok())
			break;
	}


	system->finalize();
	system->savePosesTofile("final_trajectory.txt");



	delete system;
	delete undistorter;
	delete outputWrapper;
	return 0;
}
