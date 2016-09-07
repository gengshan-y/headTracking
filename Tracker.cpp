#include "Tracker.hpp"
#include "cvLib.hpp"

// using namespace std;

unsigned int TrackingObj::getAge() {
  return age;
}   

void TrackingObj::showInfo() {
  cout << "ID\t" << ID << endl;
  cout << "age\t" << age << endl;
  cout << "pos\t(" << pos.first << "," << pos.second << ")" << endl;
  cout << "size\t" << size << endl;
  imshow("object appearance", appearance);
  pauseFrame(0);
}



unsigned int Tracker::getSize() {
  return trackingTargs.size();
}

void Tracker::pushBack(TrackingObj obj) {
  trackingTargs.push_back(obj);
}
