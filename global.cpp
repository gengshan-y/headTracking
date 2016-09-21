#include "global.hpp"

using namespace std;
using namespace cv;

/* Define a list variables */
/* for building detector */
const CvSize winSize = cvSize(64, 64);    //window size 
const CvSize blockSize = cvSize(16, 16);  //block size, fixed 
const CvSize blockStride = cvSize(8, 8);  //block stride, a multiple of cellSize 
const CvSize winStride = cvSize(8, 8);    //window stride, a multiple of blockStride 
const CvSize cellSize = cvSize(8, 8);     //cell size, fixed 
const int nbins = 9;  // number of direction bins, fixed 
