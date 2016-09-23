#include "global.hpp"
#include "opencv2/nonfree/features2d.hpp"
#include "cmpLib.hpp"

using namespace std;
using namespace cv;

/* Compare two images using HSV histogram */
double imgCmpHistHSV(Mat oriImg, Mat targImg) {
    /* Initialization */
    int h_bins = 50;
    int s_bins = 50;
    int histSize[] = {h_bins, s_bins};
    float h_ranges[] = { 0, 180 };
    float s_ranges[] = { 0, 256 };
    const float* ranges[] = { h_ranges, s_ranges };
    int channels[] = {0, 1};
    Mat hsvBaseOri, hsvBaseTarg;
    MatND histOri, histTarg;

    /* Process original detection result */
    cvtColor(oriImg, hsvBaseOri, COLOR_BGR2HSV);
    calcHist(&hsvBaseOri, 1, channels, Mat(), histOri, 2, histSize, ranges,
             true, false);
    normalize(histOri, histOri, 0 ,1, NORM_MINMAX, -1, Mat());

    /* Process target detection result */
    cvtColor(targImg, hsvBaseTarg, COLOR_BGR2HSV);
    calcHist(&hsvBaseTarg, 1, channels, Mat(), histTarg, 2, histSize, ranges,
             true, false);
    normalize(histTarg, histTarg, 0 ,1, NORM_MINMAX, -1, Mat());

    /* Compare histograms and show results */
    return compareHist(histOri, histTarg, 0);
}

/* compare two images using BGR histogram */
double imgCmpHistBGR(Mat oriImg, Mat targImg) {
    /* Initialization */
    int histSize[] = {50, 50, 50};

    float range1[] = {0, 256};
    float range2[] = {0, 256};
    float range3[] = {0, 256};
    const float* ranges[] = { range1, range2, range3 };
    int channels[] = {0, 1, 2};
    MatND histOri, histTarg;

    /* Process original detection result */
    calcHist(&oriImg, 1, channels, Mat(), histOri, 2, histSize, ranges,
             true, false);
    normalize(histOri, histOri, 0 ,1, NORM_MINMAX, -1, Mat());

    /* Process target detection result */
    calcHist(&targImg, 1, channels, Mat(), histTarg, 2, histSize, ranges,
             true, false);
    normalize(histTarg, histTarg, 0 ,1, NORM_MINMAX, -1, Mat());

    /* Compare histograms and show results */
    return compareHist(histOri, histTarg, 0);
}

/* compare two features using cosine similarity */
double cosSimilarity(vector<float> feat1, vector<float> feat2) {
    float ab = inner_product(feat1.begin(), feat1.end(), feat2.begin(), 0.0);
    // should use 0.0 instead of, otherwise will use int as accumulater
    float aa = inner_product(feat1.begin(), feat1.end(), feat1.begin(), 0.0);
    float bb = inner_product(feat2.begin(), feat2.end(), feat2.begin(), 0.0);
    return ab / sqrt(aa*bb);
}

/* compare two images using hog features */
double imgCmpHOG(Mat oriImg, Mat targImg) {
    vector<float> featVecOri, featVecTarg;
    HOGDescriptor hog(winSize, blockSize, blockStride, cellSize, nbins);
    hog.compute(oriImg, featVecOri, winStride);
    hog.compute(targImg, featVecTarg, winStride);

    // cout << cosSimilarity(featVecOri, featVecTarg) << endl;  // cosine
    return cosSimilarity(featVecOri, featVecTarg);
    // return -1 * norm(featVecOri, featVecTarg, NORM_L1);  // norm

}

/* helper function for sorting objects */
bool matchCmp(DMatch m1, DMatch m2) {
    return m1.distance < m2.distance;
}

/* Get surf descriptor of an image */
Mat getSURTDescriptor(Mat img) {
    /* convert to gray scale */
    cvtColor(img, img, CV_BGR2GRAY);

    /* detect key points */
    vector<KeyPoint> keyPoints;
    SurfFeatureDetector detector(10);  // min Hessian, smaller gives more keypoints
    detector.detect(img, keyPoints);

    //-- Draw keypoints
    Mat img_keypoints;

    drawKeypoints( img, keyPoints, img_keypoints, Scalar::all(-1), DrawMatchesFlags::DEFAULT );

    //-- Show detected (drawn) keypoints
    // imshow("Keypoints", img_keypoints);
    // waitKey(0);

    /* calculate descriptors */
    Mat descriptors;
    SurfDescriptorExtractor extractor;
    extractor.compute(img, keyPoints, descriptors);
    return descriptors;
}

/* Get SIFT descriptor of an image */
Mat getSIFTDescriptor(Mat img) {
    /* convert to gray scale */
    cvtColor(img, img, CV_BGR2GRAY);

    /* detect key points */
    vector<KeyPoint> keyPoints;
    Mat descriptors;
    SIFT detector(10, 3, 0.04, 10, 1.6);
    detector.operator()(img, Mat(), keyPoints, descriptors);

    //-- Draw keypoints
    Mat img_keypoints;
    drawKeypoints( img, keyPoints, img_keypoints, Scalar::all(-1), DrawMatchesFlags::DEFAULT );

    //-- Show detected (drawn) keypoints
    // imshow("Keypoints", img_keypoints);
    // waitKey(0);

    return descriptors;
}


/* match two images using descriptor distance */
double imgCmpDesMatch(Mat oriImg, Mat targImg) {
    // Mat descriptors_1 = getSURTDescriptor(oriImg);
    // Mat descriptors_2 = getSURTDescriptor(targImg);
    Mat descriptors_1 = getSIFTDescriptor(oriImg);
    Mat descriptors_2 = getSIFTDescriptor(targImg);

    if (descriptors_1.empty() || descriptors_2.empty()) {
        cout << "empty descriptor... return" << endl;
        return 0;
    }
 
    /* match descriptor vectors using FLANN */
    FlannBasedMatcher matcher;
    vector<DMatch> matches;
    matcher.match(descriptors_1, descriptors_2, matches);
    
    sort(matches.begin(), matches.end(), matchCmp);

    //-- Draw only "good" matches (i.e. whose distance is less than 2*min_dist,
    //-- or a small arbitary value ( 0.02 ) in the event that min_dist is very
    //-- small)
    //-- PS.- radiusMatch can also be used here.
    std::vector< DMatch > good_matches;

    for( int i = 0; i < descriptors_1.rows; i++ )
    { if( matches[i].distance <= max(double(2*(matches[0].distance)), 0.02) )
    { good_matches.push_back( matches[i]); }
    }

    //-- Draw only "good" matches
    // Mat img_matches;
    // drawMatches( oriImg, keyPoints1, targImg, keyPoints2,
    //            good_matches, img_matches, Scalar::all(-1), Scalar::all(-1),
    //            vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );

    //-- Show detected matches
    // imshow( "Good Matches", img_matches );
    // waitKey(0);

    double accu = 0;
    for ( auto it = good_matches.begin(); it != good_matches.end(); it++) {
        accu += (*it).distance;
        // cout << (*it).distance << endl;
    }

    return (-1) * accu / matches.size();
}
