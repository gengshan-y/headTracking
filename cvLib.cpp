#include <fstream>
#include <opencv2/opencv.hpp>
#include "Tracker.hpp"
#include "cvLib.hpp"
#include "cmpLib.hpp"

using namespace std;
using namespace cv;

/* Define a list variables */
/* for building detector */
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

/* Test sate parsing */
void testStateParsing(TrackingObj testObj) {
  TrackingObj tmpObj = testObj;
  tmpObj.attr2State();  // Flatten the attributes
  tmpObj.state2Attr();  // Fold the attributes
  if ( testObj == tmpObj ) {  // Make sure they are identical
      cout << "pass state parsing test" << endl;
  }
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
    // int centerX = detRes.x + detRes.width / 2;
    // int centerY = detRes.y + detRes.height / 2;

    /* Crop image from center with fixed size */
    // Rect tmpBox = Rect(centerX, centerY, 64, 64);
    Rect tmpBox = Rect(detRes.x, detRes.y, 64, 64);

    /* if exceeds boundries */
    if (tmpBox.x + 64 > targImg.cols || tmpBox.y + 64 > targImg.rows) {
        targImg(detRes).copyTo(croppedImg);
        resize(croppedImg, croppedImg, Size(64, 64));  // resize to fixed size 
        cout << "exceeded bounding box" << endl;
    }
    else {
        targImg(tmpBox).copyTo(croppedImg);  // to avoid referencing origin
    }
    return TrackingObj(currID, croppedImg, detRes);  // measured object

                                                  // current ID is a faked one
}


void updateTracker(vector<Rect> found, Mat targImg,
                   vector<TrackingObj>& tracker) {
    /* Upgrade old tracking objects */
    for (auto it = tracker.begin(); it != tracker.end(); it++) {
        (*it).incAge();
        (*it).predKalmanFilter();
        // (*it).showInfo();
    }

    /* Build measured objects */
    vector<TrackingObj> meaObjs;
    for (auto it = found.begin(); it != found.end(); it++)
        meaObjs.push_back(measureObj(targImg, *it));  // measured object
    // testStateParsing(meaObjs[0]);  // test the parsing interface

    /* Update/Add tracking objects */
    for (auto it = meaObjs.begin(); it != meaObjs.end(); it++) {
        cout << "@@comparing stage" << endl;
        /* get measured state */
        // cout << "measured..." << endl;
        // (*it).showState();
        vector<float> meaArray = (*it).getStateVec();

        vector<float> scoreArr;  // to store the comparison scores 
        for (auto itt = tracker.begin(); itt != tracker.end(); itt++) {
            /* get tracker predicted state */
            // cout << "predicted..." << endl; 
            // (*itt).showState();
            vector<float> predArray = (*itt).getStateVec();

            // compare states
            float stateScore = norm(meaArray, predArray, NORM_L1);
            float score;
            cout << "dist metric:\t" << stateScore << endl;;

            // get SVM score for measurement
            float SVMScore = (*itt).testSVM( (*it).getAppearance() );
            cout << "SVM score: \t" << SVMScore << endl;
            waitKey(0);
            // score = stateScore + SVMScore;
            score = SVMScore;

            scoreArr.push_back(score);  // add score to an array
        }

        unsigned int targIdx = distance(scoreArr.begin(),
                              max_element(scoreArr.begin(), scoreArr.end()));
        // unsigned int targIdx = distance(scoreArr.begin(),
        //                       min_element(scoreArr.begin(), scoreArr.end()));
        // if the highest score is higher than a th
        // if (scoreArr.size() != 0 && scoreArr[targIdx] < 1000) {
        if (scoreArr.size() != 0 && scoreArr[targIdx] > 0.5) {
            cout << "**ID " << tracker[targIdx].getID() << " updated" << endl;
            /* update the according tracker */
            // update SVM
            tracker[targIdx].updateSVM( targImg, (*it).getAppearance() );
            tracker[targIdx].updateKalmanFilter( (*it).getMeaState() );
            continue;
        }

        /* Else push detection results to tracker */
        // initialize SVM for *it
        (*it).initSVM(targImg);
        tracker.push_back(*it);
        currID++;  // update ID
        cout << "**ID " << tracker.back().getID() << " added." << endl;
        tracker.back().showInfo();
    }

    /* Get rid of out-dated objects */
    for (int it = tracker.size() - 1; it >= 0; it--) {
        if ( (tracker[it]).getAge() > 10 ) {
            cout << "ID " << tracker[it].getID() << " to be deleted." << endl;
            // delete SVM for it
            (*(tracker.begin() + it)).rmSVM();
            tracker.erase(tracker.begin() + it);
        }
    }
}

/* draw bounding box */
void drawBBox(vector<Rect> found, Mat& targImg) {
    for (auto it = found.begin(); it != found.end(); it++){
        Rect r = *it;
        rectangle(targImg, r.tl(), r.br(), Scalar(0, 255, 0), 3);
    }
}

void extBBox(vector<Rect>& found) {
    for (unsigned int it = 0; it < found.size(); it++) {
        Rect r = found[it];
        // the HOG detector returns slightly larger rectangles
        // so we slightly shrink the rectangles to get a nicer output
        r.x += cvRound(r.width*0.1);
        r.width = cvRound(r.width*0.9);
        r.y += cvRound(r.height*0.07);
        r.height = cvRound(r.height*0.9);
        found[it] = r;
    }
}

/* combine two identical-sized images */
Mat combImgs(Mat img1, Mat img2) {
    Size sz1 = img1.size();
    Size sz2 = img2.size();
    Mat img3(sz1.height + 40, sz1.width + sz2.width, CV_8UC3,
             cvScalar(255, 255, 255));
    img1.copyTo(img3(Rect(0, 0, sz1.width, sz1.height)));
    img2.copyTo(img3(Rect(sz1.width, 0, sz2.width, sz2.height)));
    return img3;
}
