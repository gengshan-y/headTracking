#ifndef CMP_LIB_HPP
#define CMP_LIB_HPP

using namespace cv;

/* Compare two images using HSV histogram */
double imgCmpHistHSV(Mat oriImg, Mat targImg);

/* compare two images using BGR histogram */
double imgCmpHistBGR(Mat oriImg, Mat targImg);

/* compare two features using cosine similarity */
double cosSimilarity(vector<float> feat1, vector<float> feat2);

/* compare two images using hog features */
double imgCmpHOG(Mat oriImg, Mat targImg);

/* helper function for sorting objects */
bool matchCmp(DMatch m1, DMatch m2);

/* Get surf descriptor of an image */
Mat getSURTDescriptor(Mat img);

/* Get SIFT descriptor of an image */
Mat getSIFTDescriptor(Mat img);

/* match two images using descriptor distance */
double imgCmpDesMatch(Mat oriImg, Mat targImg);

#endif  // CMP_LIB_HPP
