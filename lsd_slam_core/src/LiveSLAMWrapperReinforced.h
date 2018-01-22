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

	/** Callback function for new RGB images. */
	void newImageCallback(const cv::Mat& img, Timestamp imgTime);

	inline SlamSystemReinforced* getSlamSystem() {return monoOdometry;}

private:
	
	// monoOdometry
	SlamSystemReinforced* monoOdometry;

};

}
