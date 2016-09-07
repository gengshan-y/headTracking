#include <iostream>
#include "Tracker.hpp"
#include "cvLib.hpp"

using namespace std;
using namespace cv;

/* Define a list variables */
/* for building detector */ 
CvSize winSize = cvSize(64, 64);    //window size 
CvSize blockSize = cvSize(16, 16);  //block size, fixed 
CvSize blockStride = cvSize(8, 8);  //block stride, a multiple of cellSize 
CvSize winStride = cvSize(8, 8);    //window stride, a multiple of blockStride 
CvSize cellSize = cvSize(8, 8);     //cell size, fixed 
int nbins = 9;  // number of direction bins, fixed 
const char* detectorPath = "./HogDetector.txt";  // const char* for input file 
 
Size imgSize = Size(352, 198);  // resized image size 
char countStr [50];  // global current frame to store results

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
    unsigned int count = 0;  // initialize on the fist frame
    vector<TrackingObj> tracker;  // a tracker to monitor all heads
    // currID = 0;  // inital ID is 0

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
    for(;;) {
        count++;
        targetVid >> frame;
        if(frame.empty()) {
            break;
        }

        /* get frame progress */
        sprintf(countStr, "%04d", count);  // padding with zeros
        cout << "\nframe\t" << countStr << "/" << totalFrame << endl;
       
        /* process a frame and get detection result */
        resize(frame, frame, imgSize);  // set to same-scale as train
        hog.detectMultiScale(frame, found, 0, winStride, Size(0, 0), 1.05, 3);

        /* remove inner boxes */
        found = rmInnerBoxes(found);

        /* get cropped images in detRests */
        updateTracker(found, frame, tracker);

        cout << tracker.size() << endl;

        // countRests(detRests);  // count detection results counting

        /* save cropped image */
        // svCroppedImg(found, frame);

        /* draw bounding box */
        drawBBox(found, frame);
    
        /* counting */
        // countHead(detRests, frame, memoHead, up, down);

        /* put information on image */
        putText(frame, 
                "frame " + string(countStr) + "/" + to_string(totalFrame), 
                cvPoint(20, 20), FONT_HERSHEY_COMPLEX_SMALL,
                0.8, cvScalar(0, 0, 0));  // frame progress

        /* show detection result */
        if (string(argv[2]) == "y") {
            imshow("demo", frame);
            pauseFrame(1);
        }
    }
    
    return 0;
}
