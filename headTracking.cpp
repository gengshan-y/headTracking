#include <iostream>
#include "Tracker.hpp"

using namespace std;
using namespace cv;

int main(int argc, char* argv[]) {

  Tracker tracker;
  for (unsigned int it = 0; it < 5; it++) {
    Mat testImg(64, 64, CV_8UC3, cvScalar(255, 0, 0));
    Rect testBBox(2, 2, 64, 64);
    tracker.trackingTargs.push_back(TrackingObj(0, testImg, testBBox));
  }

  cout << Tracker.getSize() << endl;

  // test.showInfo(); 

  return 0;
}
