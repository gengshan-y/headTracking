#include <fstream>
#include <opencv2/opencv.hpp>
#include "Tracker.hpp"
#include "cvLib.hpp"

using namespace std;
using namespace cv;

/* Define a list of variables */
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

void updateTracker(vector<Rect> found, Mat targImg,
                   vector<TrackingObj>& tracker) {

    Mat croppedImg;  // detected head img
    for (auto it = found.begin(); it != found.end(); it++) {
        /* get the center */
        int centerX = (*it).x + (*it).width / 2;
        int centerY = (*it).y + (*it).height / 2;

        /* Crop image from center with fixed size */
        Rect tmpBox = Rect(centerX, centerY, 64, 64);
        /* if exceeds the boundries */
        if (tmpBox.x + 64 > targImg.cols || tmpBox.y + 64 > targImg.rows) {
            croppedImg = targImg(*it);
            resize(croppedImg, croppedImg, Size(64, 64));  // resize to fixed size 
        }
        else {
            croppedImg = targImg(tmpBox);
        }

        /* Push detection results to tracker */
        tracker.push_back(TrackingObj(currID, croppedImg, *it));
        currID++;  // update ID
    }

    /* Get rid of out-dated objects */
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
