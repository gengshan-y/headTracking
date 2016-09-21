#ifndef GLOBAL_HPP
#define GLOBAL_HPP

#include <opencv2/opencv.hpp>

/* Define interface for global vars */
extern const CvSize winSize;    //window size 
extern const CvSize blockSize;  //block size, fixed 
extern const CvSize blockStride;  //block stride, a multiple of cellSize 
extern const CvSize winStride;    //window stride, a multiple of blockStride 
extern const CvSize cellSize;     //cell size, fixed 
extern const int nbins;  // number of direction bins, fixed 

#endif  // GLOBAL_HPP
