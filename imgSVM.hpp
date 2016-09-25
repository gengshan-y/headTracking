#ifndef IMG_SVM
#define IMG_SVM

#include <opencv2/opencv.hpp>
#include <fstream>
#include "global.hpp"

using namespace cv;


/** The class for pasing images to SVM classifier
***/
class imgSVM {
 public:
  /* Constructor */
  imgSVM() : featSize(1764), trainDataMat(0, 1764, CV_32FC1),
             labelsMat(0, 1, CV_32FC1){
  }

  /* Show basic information of SVM classifier */
  void showInfo();

  /* Read images to a vector of mats */
  vector<Mat> path2img(char* imgListPath);

  /* From a vector of image mats to features */
  Mat img2feat(vector<Mat> imgVec);

  /* Read images in the list and compute features */
  Mat path2feat(char* imgListPath);

  /* Parse Mat into training sample */
  void Mat2samp();

  /* Configure parameters */
  void SVMConfig();

  /* Training SVM */
  void SVMTrain();

  /* Predicing label */
  float SVMPredict(Mat sampleMat);

  /* Push training data */
  void fillData(Mat trainPos, Mat trainNeg);

  /* Get feature size of the classifier */
  unsigned int getFeatSize();

 private:
  CvSVM SVM;  // use pointer to avoid private CvSVM copy constructor
  CvSVMParams params;
  unsigned int featSize;
  Mat trainDataMat;
  Mat labelsMat;
  
  // friend class TrackingObj;
};

#endif  // IMG_SVM
