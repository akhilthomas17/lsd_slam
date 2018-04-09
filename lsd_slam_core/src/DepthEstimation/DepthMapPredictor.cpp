#include "DepthEstimation/DepthMapPredictor.h"

using namespace lsd_slam;

void DepthMapPredictor::createKeyframeManager(Frame* new_keyframe)
{
  createKeyFrame(new_keyframe);
}
