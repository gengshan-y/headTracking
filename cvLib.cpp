#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

/* Pause current frame */
void pauseFrame(unsigned int milliSeconds) {
    char key = (char) waitKey(milliSeconds);
    switch (key) {
    case 'j':
        return;
    }
}
