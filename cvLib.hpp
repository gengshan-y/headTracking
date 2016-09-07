#ifndef CV_LIB_HPP
#define CV_LIB_HPP

/* Declare a list of variables */
extern unsigned int currID;  // current object ID, declare with extern and 
                             // define in .cpp to avoid multiple definition

/* Pause current frame */
void pauseFrame(unsigned int milliSeconds);

/* Build detecotr from training result */
void buildDetector(HOGDescriptor& hog, const char* detectorPath);

/* remove inner boxes */
vector<Rect> rmInnerBoxes(vector<Rect> found);

/* get cropped images from frame */
void updateTracker(vector<Rect> found, Mat targImg, 
                   vector<TrackingObj>& tracker);

/* draw bounding box */
void drawBBox(vector<Rect> found, Mat& targImg);

#endif /* CV_LIB_HPP */
