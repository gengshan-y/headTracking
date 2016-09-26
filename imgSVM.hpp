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

  /* Configure parameters */
  void SVMConfig();

  /* Training SVM */
  void SVMTrain();

  /* Predict single score */
  float SVMPredict(Mat sampleMat);

  /* Predict multiple labels */
  void SVMPredict(Mat sampleMat, Mat& res);

  /* Push training data */
  void fillData(Mat trainPos, Mat trainNeg);

  /* Get feature size of the classifier */
  unsigned int getFeatSize();

 private:
  CvSVM SVM; 
  CvSVMParams params;
  unsigned int featSize;
  Mat trainDataMat;
  Mat labelsMat;
};

#endif  // IMG_SVM
