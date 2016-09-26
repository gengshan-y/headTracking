# headTracking

C++ implementation for multi-target tracking algorithm from paper *Multi-target tracking by learning local-to-global trajectory models*.  
Using Kalman filter and SVM classifier to generate tracking scores.

## Note
- **imgSVM** class trains/tests SVM classifier through images.  
- **Tracker** is a higher level class including implementation of Kalman Filters and call of imgSVM class.  
- **cvLib** implements methods for updating trackers and for other utilities specific for tracking task.
- **cmpLib** implements methods of image feature extraction and comparison.
- global.hpp/.cpp keeps intersection of global variables for both imgSVM class and tracking task.

## Usage
- `./headTracking /data/gengshan/vid/testMultiTarget.avi y`

## todo
- use distance to leverage svm score...
- divide updateTracker function
- optimzie display setting... put tracking result in detection result later. 

## Log
- params: negNum in TrackingObj class.
