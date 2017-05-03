#include "kalman_filter.h"
#include <iostream>
#include <assert.h>

using Eigen::MatrixXd;
using Eigen::VectorXd;

KalmanFilter::KalmanFilter() {}

KalmanFilter::~KalmanFilter() {}

void KalmanFilter::Init(VectorXd &x_in, MatrixXd &P_in, MatrixXd &F_in,
                        MatrixXd &H_in, MatrixXd &R_in, MatrixXd &Q_in) {
  x_ = x_in;
  P_ = P_in;
  F_ = F_in;
  H_ = H_in;
  R_ = R_in;
  Q_ = Q_in;
}

void KalmanFilter::Predict() 
{
  // predict the state
  x_ = F_ * x_;
  P_ = F_ * P_ * F_.transpose() + Q_;
}

void KalmanFilter::Update(const VectorXd &z) 
{
  // update the state by using Kalman Filter equations (used for LiDAR)

  Eigen::MatrixXd y_ = MatrixXd(2, 1);
  y_ = z - H_ * x_;

  Eigen::MatrixXd S_ = MatrixXd(2, 2);
  S_ = H_ * P_ * H_.transpose() + R_;

  Eigen::MatrixXd K_ = MatrixXd(4, 2);
  K_ = P_ * H_.transpose() * S_.inverse();

  x_ = x_ + K_ * y_;

  Eigen::MatrixXd I_ = MatrixXd(4, 4);
  I_ << 1, 0, 0, 0,
        0, 1, 0, 0,
        0, 0, 1, 0,
        0, 0, 0, 1; 

  P_ = (I_ - K_ * H_) * P_;

}

void KalmanFilter::UpdateEKF(const VectorXd &z) 
{
  // update the state by using Extended Kalman Filter equations (used for radar)

    float px = x_(0);
    float py = x_(1);
    float vx = x_(2);
    float vy = x_(3);

    float c_1 = sqrt(px*px + py*py);
    float c_2 = atan2(py, px);
    float c_3 = (px * vx + py * vy) / c_1;

    // if c_1 is zero, set c_3 to zero in order to avoid dividing by zero
    if(c_1 == 0){
      std::cout << "px^2 + py^2 is zero" << std::endl;
      c_3 = 0;
    }

    // populate the h(x) function for the extended Kalman Filter
    Eigen::MatrixXd h_x = MatrixXd(3, 1);
    h_x << c_1,
           c_2,
           c_3;

    Eigen::MatrixXd y_ = MatrixXd(3, 1);
    y_ = z - h_x;

    // Ensure that the value of phi in y_ is between -pi and pi.
    double pi = 3.1415926535897932;

    if(y_(1) < -pi){
    	while(y_(1) < -pi)
    		y_(1) = y_(1) + 2*pi;
    	assert(y_(1) > -pi && y_(1) < pi);
    }

    if(y_(1) > pi){
    	while(y_(1) > pi)
    		y_(1) = y_(1) - 2*pi;
    	assert(y_(1) > -pi && y_(1) < pi);
    }

    Eigen::MatrixXd S_ = MatrixXd(3, 3);
    S_ =  H_ * P_ * H_.transpose() + R_;

    Eigen::MatrixXd K_ = MatrixXd(4, 3);
    K_ = P_ * H_.transpose() * S_.inverse();

    x_ = x_ + K_ * y_;

    Eigen::MatrixXd I_ = MatrixXd(4, 4);
    I_ << 1, 0, 0, 0,
        0, 1, 0, 0,
        0, 0, 1, 0,
        0, 0, 0, 1; 

    P_ = (I_ - K_ * H_) * P_;

}
