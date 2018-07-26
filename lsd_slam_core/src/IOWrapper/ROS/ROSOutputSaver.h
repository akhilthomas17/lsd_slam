#pragma once

#include "ROSOutput3DWrapper.h"

namespace lsd_slam
{

class ROSOutputSaver: public ROSOutput3DWrapper
{
	public:
		ROSOutputSaver(int width, int height, int newItrNum, std::string newResultFolder);
		ROSOutputSaver(int width, int height);
		~ROSOutputSaver();

		// saves the current keyframe images to file to use as training data
		virtual void publishKeyframe(Frame* f);

	private:
		int itrNum;
		bool debug;
		std::string resultFolder;
};

}