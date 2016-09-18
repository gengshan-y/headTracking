#ifndef CV_LIB_HPP
#define CV_LIB_HPP

/* Define interface for global vars */
extern const CvSize winSize;    //window size 
extern const CvSize blockSize;  //block size, fixed 
extern const CvSize blockStride;  //block stride, a multiple of cellSize 
extern const CvSize winStride;    //window stride, a multiple of blockStride 
extern const CvSize cellSize;     //cell size, fixed 
extern const int nbins;  // number of direction bins, fixed 
extern const char* detectorPath;  // const char* for input file 

extern const Size imgSize;  // resized image size 

extern char countStr [50];  // global current frame to store results
extern unsigned int currID;  // current object ID, declare with extern and 
                             // define in .cpp to avoid multiple definition

/* Pause current frame */
void pauseFrame(unsigned int milliSeconds);

/* Build detecotr from training result */
void buildDetector(HOGDescriptor& hog, const char* detectorPath);

/* remove inner boxes */
vector<Rect> rmInnerBoxes(vector<Rect> found);

/* build tracking object based on a detection result */
TrackingObj measureObj(Mat targImg, Rect detRes);

/* get cropped images from frame */
void updateTracker(vector<Rect> found, Mat targImg, 
                   vector<TrackingObj>& tracker);

/* draw bounding box */
void drawBBox(vector<Rect> found, Mat& targImg);

/*  */

#endif /* CV_LIB_HPP */
