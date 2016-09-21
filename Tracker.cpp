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
  cout << "vel\t(" << vel.first << "," << vel.second << ")" << endl;
  cout << "size\t" << size << endl;
  imshow("object appearance", appearance);
  pauseFrame(0);
}

void TrackingObj::attr2State() {
  int count = 0;

  state.at<float>(count++, 0) = pos.first;
  state.at<float>(count++, 0) = pos.second;
  state.at<float>(count++, 0) = vel.first;
  state.at<float>(count++, 0) = vel.second;
  state.at<float>(count++, 0) = size;

  /* Assert */
  if(count != state.rows) {
    cout << "Size check failed." << endl;
    exit(-1);
  }
}

void TrackingObj::state2Attr() {
  pos = make_pair(state.at<float>(0, 0), state.at<float>(1, 0));
  vel = make_pair(state.at<float>(2, 0), state.at<float>(3, 0));
  size = state.at<float>(4, 0);
}

bool TrackingObj::operator==(const TrackingObj& other) {

  if(pos != other.pos) {
    cout << "position not matched" << endl;
    return false;
  }

  if(vel != other.vel) {
    cout << "velocity not matched" << endl;
    return false;
  }

  if(size != other.size) {
    cout << "size not matched" << endl;
    return false;
  }

  return true;
}

void TrackingObj::showState() {
  cout << "state: " << endl;
  for (int it = 0; it < state.rows; it++) {
    cout << state.at<float>(it, 0) << endl;
  }
}

void TrackingObj::initKalmanFilter() {
  attr2State();
  showState();
  KF.transitionMatrix = *( Mat_<float>(5, 5) << 1, 0, 1, 0, 0,  // fps-1 
                                                0, 1, 0, 1, 0,
                                                0, 0, 1, 0, 0,
                                                0, 0, 0, 1, 0,
                                                0, 0, 0, 0, 1 );

  KF.measurementMatrix = *( Mat_<float>(3, 5) << 1, 0, 0, 0, 0,
                                                 0, 1, 0, 0, 0,
                                                 0, 0, 0, 0, 1 );

  setIdentity(KF.processNoiseCov, Scalar::all(1e-5));   // to be tuned
  setIdentity(KF.measurementNoiseCov, Scalar::all(1e-1));  // to be tuned

  setIdentity(KF.errorCovPost, Scalar::all(1));
  KF.statePost = state;

  cout << "transition matrix:\n" << KF.transitionMatrix << endl;
  cout << "measurement matrix:\n" << KF.measurementMatrix << endl;
  cout << "process noise:\n" << KF.processNoiseCov << endl;
  cout << "measurement noise\n:" << KF.measurementNoiseCov << endl;
  cout << "initial state covariance:\n" << KF.errorCovPost << endl;
  cout << "initial state:\n" << KF.statePost << endl;
}

void TrackingObj::refreshKalmanFilter() {
  Mat prediction = KF.predict();
  cout << "predicted state:\n" << prediction << endl;
  cout << "prior state:\n" << KF.statePre << endl;
  cout << "prior state covariance:\n" << KF.errorCovPre << endl;

  /* correct using new measurement */
  exit(-1);
}
