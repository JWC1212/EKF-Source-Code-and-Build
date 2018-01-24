#include <iostream>
#include "tools.h"

using Eigen::VectorXd;
using Eigen::MatrixXd;
using std::vector;

Tools::Tools() {}

Tools::~Tools() {}

VectorXd Tools::CalculateRMSE(const vector<VectorXd> &estimations,
                              const vector<VectorXd> &ground_truth) {
    VectorXd rmse(4);
    rmse << 0.0, 0.0, 0.0, 0.0;
    if ((estimations.size()!=ground_truth.size())||(estimations.size()==0)){
        cout << "size is non-equal or zero. " <<endl;
        return rmse;
    }
    for (unsigned i=0;i<estimations.size();i++){
        VectorXd residual(4);
        residual = estimations[i] - ground_truth[i];
        residual = residual.array()*residual.array();
        rmse += residual;
    }
    rmse = rmse/estimations.size();
    rmse = rmse.array().sqrt();
    return rmse;
}

MatrixXd Tools::CalculateJacobian(const VectorXd& x_state) {
    float px = x_state(0);
    float py = x_state(1);
    float vx = x_state(2);
    float vy = x_state(3);
    
    float sqr = px*px + py*py;
    float sqroot = sqrt(sqr);
    
    MatrixXd Hj(3,4);
    
    if (fabs(sqroot) < 0.000001){
        cout << "both X and Y are zero."<<endl;
        return Hj;
    }
    
    Hj<< px/sqroot,py/sqroot, 0.0, 0.0,
        -(py/sqr), px/sqr,0.0,0.0,
    py*(vx*py - vy*px)/(sqr*sqroot), px*(px*vy-py*vx)/(sqr*sqroot),px/sqroot,py/sqroot;
    
    return Hj;
}
