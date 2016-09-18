#include "Tracker.hpp"
#include "cvLib.hpp"

// using namespace std;

unsigned int TrackingObj::getAge() {
  return age;
}   

unsigned int TrackingObj::getID() {
  return ID;
}

void TrackingObj::incAge() {
  age++;
  cout << "age increased." << endl; 
}

void TrackingObj::showInfo() {
  cout << "ID\t" << ID << endl;
  cout << "age\t" << age << endl;
  cout << "pos\t(" << pos.first << "," << pos.second << ")" << endl;
  cout << "size\t" << size << endl;
  imshow("object appearance", appearance);
  pauseFrame(0);
}

void TrackingObj::flattenAttr() {
  unsigned int count = 0;

  /* For appearance */
  // cout << appearance.type() << endl;  // check mat type
  Mat appFloat;
  appearance.convertTo(appFloat, CV_32FC3, 1.0/255);
  // convert to float to be compatible with other parameters
  float* p;
  for (int it = 0; it < appFloat.rows; it++) {
    p = appFloat.ptr<float>(it);
    for (int itt = 0; itt < appFloat.cols * 3; itt++) {  // for 3 channels
      params[count++] = p[itt];
    }
  }
  
  /* For position and size */
  params[count++] = pos.first;
  params[count++] = pos.second;
  params[count++] = size;

  /* For velocity and accel */

  /* Assert */
  if(count != params.size()) {
    cout << "Size check failed." << endl;
    exit(-1);
  }
}

void TrackingObj::foldParams() {
  unsigned int count = 0;

  /* For appearance */
  Mat appFloat(appearance.rows, appearance.cols, CV_32FC3);
  float* p;
  for (int it = 0; it < appFloat.rows; it++) {
    p = appFloat.ptr<float>(it);
    for (int itt = 0; itt < appFloat.cols * 3; itt++) {
      p[itt] = params[count++];
    }
  }
  appFloat.convertTo(appearance, 16, 255);  // 16 is CV_8UC3
  // convert to original type
  
  /* For position and size */
  float posX = params[count++];
  float posY =  params[count++];
  pos = make_pair(posX, posY);
  size = params[count++];

  /* Assert */
  if(count != params.size()) {
    cout << "Size check failed." << endl;
    exit(-1);
  }
}

bool TrackingObj::operator==(const TrackingObj& other) {
  if (ID != other.ID) {
    cout << "ID not matched" << endl;
    return false;
  }
  if (age != other.age) {
    cout << "age not matched" << endl;
    return false;
  }
  
  if (appearance.rows != other.appearance.rows || 
      appearance.cols != other.appearance.cols) {
    cout << "appearance size not matched" << endl;
    return false;
  }
  unsigned int* p;
  unsigned int* q;
  unsigned int accum = 0;  // accumulate the error
  Mat otherAppearance = other.appearance;
  for (int it = 0; it < appearance.rows; it++) {
    p = appearance.ptr<unsigned int>(it);
    q = otherAppearance.ptr<unsigned int>(it);
    for (int itt = 0; itt < appearance.cols * 3; itt++) {
      accum += abs(p[itt] - q[itt]);
      // cout << "row " << it << "\tcol " << itt << "\t" 
      //      << p[itt] << "\t" <<  q[itt] << endl;
    }
  }
  cout << "accumulated L1 error: " << accum << endl;
  if (accum != 0) {
    cout << "appearance not matched" << endl;
    return false;  
  }

  if (pos.first != other.pos.first || pos.second != other.pos.second) {
    cout << "position not matched" << endl;
    return false;
  }

  if (size != other.size) {
    cout << "size not matched" << endl;
    return false;
  }
  return true;  
}
