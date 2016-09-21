#include <fstream>
#include <opencv2/opencv.hpp>
#include "Tracker.hpp"
#include "cvLib.hpp"

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
const char* detectorPath = "./HogDetector.txt";  // const char* for input file 

/* for resizing image */
const Size imgSize = Size(352, 198);  // resized image size 

/* global current frame to store results */
char countStr [50];
unsigned int currID = 0;

/* Pause current frame */
void pauseFrame(unsigned int milliSeconds) {
    char key = (char) waitKey(milliSeconds);
    switch (key) {
    case 'q':
    case 'Q':
    case 27:
        exit(0);  // stop program=
    default:
        return;  // go on
    }
}

/* Build detecotr from training result */
void buildDetector(HOGDescriptor& hog, const char* detectorPath) {
    /* Loading detector */
    /* loading file*/
    ifstream detFile(detectorPath, ios::binary);
    if (!detFile.is_open()) {
        cout << "HogDetector open failed." << endl;
        exit(-1);
    }
    detFile.seekg(0, ios_base::beg);

    vector<float> x;  // for constructing SVM
    float tmpVal = 0.0f;
    while (!detFile.eof()) {
        detFile >> tmpVal;
        x.push_back(tmpVal);
    }
    detFile.close();
    // cout << x.size() << " paramters loaded." << endl;

    /* constructing detector*/
    hog.setSVMDetector(x);
}

/* remove inner boxes */
vector<Rect> rmInnerBoxes(vector<Rect> found) {
    /* empty result */
    if (found.size() == 0) {
        return found;
    }

    /* with non-empty result */
    vector<Rect> foundFiltered;
    auto it = found.begin();
    auto itt = found.begin();
    for (it = found.begin(); it != found.end(); it++) {
        for (itt = found.begin(); itt != found.end(); itt++) {
            if (it != itt && ((*it & *itt) == *it) ) {
                break;
            }
        }
        if (itt == found.end()) {
            foundFiltered.push_back(*it);
        }
    }
    return foundFiltered;
}

TrackingObj measureObj(Mat targImg, Rect detRes) {
    Mat croppedImg;  // detected head img

    /* get the center */
    int centerX = detRes.x + detRes.width / 2;
    int centerY = detRes.y + detRes.height / 2;

    /* Crop image from center with fixed size */
    Rect tmpBox = Rect(centerX, centerY, 64, 64);

    /* if exceeds boundries */
    if (tmpBox.x + 64 > targImg.cols || tmpBox.y + 64 > targImg.rows) {
        targImg(detRes).copyTo(croppedImg);
        resize(croppedImg, croppedImg, Size(64, 64));  // resize to fixed size 
    }
    else {
        targImg(tmpBox).copyTo(croppedImg);  // to avoid referencing origin
    }
    return TrackingObj(currID, croppedImg, detRes);  // measured object
                                                  // current ID is a faked one
}


void updateTracker(vector<Rect> found, Mat targImg,
                   vector<TrackingObj>& tracker) {


    /* Upgrade old object */
    for (auto it = tracker.begin(); it != tracker.end(); it++) {
        (*it).incAge();
        // (*it).showInfo();
    }

    /* Update/Add objects */
    for (auto it = found.begin(); it != found.end(); it++) {

        /* Build measured object */
        TrackingObj measuredObj = measureObj(targImg, *it);  // measured object
        
        /* update existing tracker */
        for (auto itt = tracker.begin(); itt != tracker.end(); itt++) {
            TrackingObj tmpObj = (*itt);
            /* Flatten the attributes */
            (*itt).attr2State();
            /* Fold the attributes */
            (*itt).state2Attr();
            /* Make sure they are identical */
            if ( (*itt) == tmpObj ) {
                cout << "identical" << endl;
            }
            
            /* Build a inner class for kalman object in trackign obj */
            (*itt).initKalmanFilter();
            // (*itt).refreshKalmanFilter();

            /*  */
            /* compare with current tracking objects */
        }
        if (0) {
            /* set the current tracker */
            continue;
        }

        /* Else push detection results to tracker */
        tracker.push_back(measuredObj);
        currID++;  // update ID
        cout << "ID " << tracker.back().getID() << " added." << endl;
    }

    /* Get rid of out-dated objects */
    for (int it = tracker.size() - 1; it >= 0; it--) {
        if ( (tracker[it]).getAge() > 10 ) {
            cout << "ID " << tracker[it].getID() << " to be deleted." << endl;
            tracker.erase(tracker.begin() + it);
        }
    }
}

/* draw bounding box */
void drawBBox(vector<Rect> found, Mat& targImg) {
    for (auto it = found.begin(); it != found.end(); it++){
        Rect r = *it;
        // the HOG detector returns slightly larger rectangles
        // so we slightly shrink the rectangles to get a nicer output
        r.x += cvRound(r.width*0.1);
        r.width = cvRound(r.width*0.9);
        r.y += cvRound(r.height*0.07);
        r.height = cvRound(r.height*0.9);
        rectangle(targImg, r.tl(), r.br(), Scalar(0, 255, 0), 3);
    }
}
