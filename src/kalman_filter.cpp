#include "kalman_filter.h"

using Eigen::MatrixXd;
using Eigen::VectorXd;

// Please note that the Eigen library does not initialize 
// VectorXd or MatrixXd objects with zeros upon creation.

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

void KalmanFilter::Predict() {
    x_ = F_*x_;
    P_ = F_*P_*F_.transpose() + Q_;
}

void KalmanFilter::Update(const VectorXd &z) {
    VectorXd y = z - H_*x_;
    MatrixXd S = H_*P_*H_.transpose() + R_;
    MatrixXd K = P_*H_.transpose()*S.inverse();
    
    x_ = x_ + K*y;
    long x_size = x_.size();
    P_ = (MatrixXd::Identity(x_size,x_size) - K*H_)*P_;
}

void KalmanFilter::UpdateEKF(const VectorXd &z) {
    VectorXd h(3);
    float px = x_(0);
    float py = x_(1);
    float vx = x_(2);
    float vy = x_(3);
    float sqr = px*px + py*py;
    h << sqrt(sqr), atan2(py,px),(px*vx+py*vy)/sqrt(sqr);
    
    VectorXd y = z - h;
    //Normalization phi
    if (y(1) > PI)
        y(1) = y(1) - 2*PI;
    else if (y(1) < -PI)
        y(1) = y(1) + 2*PI;
    
    MatrixXd S = H_*P_*H_.transpose() + R_;
    MatrixXd K = P_*H_.transpose()*S.inverse();
    
    x_= x_ + K*y;
    long x_size = x_.size();
    P_ = (MatrixXd::Identity(x_size,x_size) - K*H_)*P_;
}
