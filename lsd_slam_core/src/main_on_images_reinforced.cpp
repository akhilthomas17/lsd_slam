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
#include "IOWrapper/ROS/rosReconfigure.h"

#include "util/Undistorter.h"
#include <ros/package.h>

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


	// make output wrapper. just set to zero if no output is required.
	Output3DWrapper* outputWrapper = new ROSOutput3DWrapper(w,h);


	// make slam system
	SlamSystemReinforced* system = new SlamSystemReinforced(w, h, K, doSlam);
	system->init(w, h, K);
	system->setVisualization(outputWrapper);



	// open image rgb_files: first try to open as file.
	std::string source;
	std::vector<std::string> rgb_files, depth_files;
	// For reading timestamps from file
	std::vector<double> timestamps;

	if(!ros::param::get("~files", source))
	{
		printf("need source rgb_files! (set using _files:=FOLDER)\n");
		exit(0);
	}
	ros::param::del("~files");

	/*
	if(getdir(source, rgb_files) >= 0)
	{
		printf("found %d image rgb_files in folder %s!\n", (int)rgb_files.size(), source.c_str());
	}
	else */if(getFileRgbdTUM(source, rgb_files, depth_files, timestamps) >= 0)
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
        printf("Type depth image: %d\n", depthImg.type());
        depthImg.convertTo(depthImg, CV_32F, 0.0002);

		if(runningIDX == 0){
			//system->randomInit(image.data, fakeTimeStamp, runningIDX);
            cv::imshow("depth0", depthImg);
            //cv::imshow("depth1", depthImg);
            cv::waitKey(0);
            system->gtDepthInit(&image, &depthImg, fakeTimeStamp, runningIDX);
        } else
        	system->trackFrame(&image, &depthImg, runningIDX, hz == 0, fakeTimeStamp);
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
	system->savePosesTofile("lsd_slam_poses.txt");



	delete system;
	delete undistorter;
	delete outputWrapper;
	return 0;
}
