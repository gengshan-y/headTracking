#include <iostream>
#include "global.hpp"
#include "Tracker.hpp"
#include "cvLib.hpp"

using namespace std;
using namespace cv;

int main(int argc, char* argv[]) {
    /* Basic info */
    if (argc != 3) {
        cout << "hogHeadDet input-vid-path " 
             << "display-result[y/n]" << endl;
        exit(-1);
    }
    cout << "OpenCV version " << CV_VERSION << endl;

    /* Initialization */
    Mat frame;  // to store video frames
    vector<Rect> found;  // to store detection results
    unsigned int count = 160;  // initialize the fist frame to be decoded, 80
    vector<TrackingObj> tracker;  // a tracker to monitor all heads

    /* Build detector */
    HOGDescriptor hog(winSize, blockSize, blockStride, cellSize, nbins);
    buildDetector(hog, detectorPath);

    /* Read in frames and process */
    VideoCapture targetVid(argv[1]);
    if(!targetVid.isOpened()) {
        cout << "open failed." << endl;
        exit(-1);
    }
    unsigned int totalFrame = targetVid.get(CV_CAP_PROP_FRAME_COUNT);

    /* Set current to a pre-defined frame */
    targetVid.set(CV_CAP_PROP_POS_FRAMES, count);

    for(;;) {
        count++;
        targetVid >> frame;
        if(frame.empty()) {
            break;
        }

        /* get frame progress */
        sprintf(countStr, "%04d", count);  // padding with zeros
        cout << "------------------------------------------" 
             << "------------------------------------------" << endl;
        cout << "frame\t" << countStr << "/" << totalFrame << endl;
       
        /* process a frame and get detection result */
        resize(frame, frame, imgSize);  // set to same-scale as train
        hog.detectMultiScale(frame, found, 0, winStride, Size(0, 0), 1.05, 3);

        /* remove inner boxes */
        found = rmInnerBoxes(found);

        /* extend bounding box */
        extBBox(found);

        Mat dispFrame;
        frame.copyTo(dispFrame);
        /* draw bounding box */
        drawBBox(found, dispFrame);
        /* put information on image */
        putText(dispFrame, 
                "frame " + string(countStr) + "/" + to_string(totalFrame), 
                cvPoint(20, 20), FONT_HERSHEY_COMPLEX_SMALL,
                0.8, cvScalar(0, 0, 0));  // frame progress


        /* Display image */
        // imshow("demo", dispFrame);
        // pauseFrame(1);

        /* get cropped images in detRests */
        updateTracker(found, frame, tracker);
 
        // countRests(detRests);  // count detection results counting

        /* save cropped image */
        // svCroppedImg(found, frame);
    
        /* counting */
        // countHead(detRests, frame, memoHead, up, down);

        putText(dispFrame, "up: " + to_string(upAccum) + 
                        " down: " + to_string(downAccum),
                cvPoint(20, 40), FONT_HERSHEY_COMPLEX_SMALL,
                0.8, cvScalar(0, 0, 0));  // up / down counting

        /* show detection result */
        if (string(argv[2]) == "y") {
            imshow("demo", dispFrame);
            pauseFrame(1);
        }
    }
    
    return 0;
}
