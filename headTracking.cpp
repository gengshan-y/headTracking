#include <iostream>
#include "Tracker.hpp"

using namespace std;
using namespace cv;

int main(int argc, char* argv[]) {

  vector<TrackingObj> tracker;
  for (unsigned int it = 0; it < 5; it++) {
    Mat testImg(64, 64, CV_8UC3, cvScalar(255, 0, 0));
    Rect testBBox(2, 2, 64, 64);
    tracker.push_back(TrackingObj(0, testImg, testBBox));
  }

  cout << tracker.size() << endl;

  // test.showInfo(); 
  return 0;
}
