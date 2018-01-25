#pragma once
#include <LiveSLAMWrapper.h>

namespace lsd_slam
{

class SlamSystemReinforced;
class LiveSLAMWrapperROS;
class InputImageStream;
class Output3DWrapper;


struct LiveSLAMWrapperReinforced : public LiveSLAMWrapper
{
friend class LiveSLAMWrapperROS;
public:
	LiveSLAMWrapperReinforced(InputImageStream* imageStream, Output3DWrapper* outputWrapper);
	~LiveSLAMWrapperReinforced();

	/** Resets everything, starting the odometry from the beginning again. */
	void resetAll();

	/** Loop function that keeps on looking for new images and feeds it to the SLAM System */
	void Loop();

	/** Callback function for new RGBD images. */
	void newImageCallback(cv::Mat& rgbImg, Timestamp imgTime, cv::Mat&depthImg, Timestamp depthTime);

	inline SlamSystemReinforced* getSlamSystem() {return monoOdometry;}

private:
	
	// monoOdometry
	SlamSystemReinforced* monoOdometry;

};

}
