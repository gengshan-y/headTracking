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
  TrackingObj(unsigned int objID, Mat objAppearance, Rect bBox)
              : age(1), params(12291, 0) {
    ID = objID;
    appearance = objAppearance;

    /* convert rectangle to position and size */
    pos = make_pair(bBox.x + bBox.width / 2.0, bBox.y + bBox.height / 2.0);
    size = float(bBox.width * bBox.height);
  }

  /* Get the age of object */
  unsigned int getAge();

  /* Get the ID of the object */
  unsigned int getID();

  /* Increase the age of an object */
  void incAge();

  /* Print object information */
  void showInfo();

  /* Flatten attributes to a vector of parameters */
  void flattenAttr();

  /* Fold parameters to attributes */
  void foldParams();

  /* Compare attributes with another trackingObj */
  bool operator==(const TrackingObj& other);

 private:
  unsigned int ID;
  unsigned int age;  // object's existing time
  Mat appearance;  // image of detected object
  pair<float, float> pos;  // center of detected object
  float size;  // size of detected object
  vector<float> params;  // flatterned parameters, 64*64*3+2+1 dim
  // velocity and accel
};

#endif  // TRACKER_HPP
