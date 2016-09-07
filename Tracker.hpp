#ifndef TRACKER_HPP
#define TRACKER_HPP

#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

/** The class for tracking object
***/
class TrackingObj {
 public:
  /* Constructor */
  TrackingObj(unsigned int objID, Mat objAppearance, Rect bBox) : age(1) {
    ID = objID;
    appearance = objAppearance;

    /* convert rectangle to position and size */
    pos = make_pair(bBox.x + bBox.width / 2.0, bBox.y + bBox.height / 2.0);
    size = double(bBox.width * bBox.height);
  }

  /* Get the age of object */
  unsigned int getAge();

  /* Print object information */
  void showInfo();

 private:
  unsigned int ID;
  unsigned int age;  // object's existing time
  Mat appearance;  // image of detected object
  pair<double, double> pos;  // center of detected object
  double size;  // size of detected object
};

/** The class for tracker
***/
class Tracker {
 public:
  /* Constructor */
  Tracker();
 
  /* Check how many objects are stored in the tracker */
  unsigned int getSize();

  /* Insert an object on top */
  void pushBack();
 
 private:
  vector<TrackingObj> trackingTargs;
};

#endif  // TRACKER_HPP
