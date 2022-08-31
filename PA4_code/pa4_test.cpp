#include <iostream>
#include <cmath>
#include <vector>

using namespace std;

struct CURVE_FITTING_COST
{
    CURVE_FITTING_COST(double x, double y) : _x(x), _y(y) {}

    // 残差的计算
    bool operator()(
        const double *const abc, // 模型参数，有3维
        double *residual
        ) const
    {
        // residual[0] = double(_y) - ceres::exp(abc[0] * double(_x) * double(_x) + abc[1] * double(_x) + abc[2]); // y-exp(ax^2+bx+c)
        residual[0] = double(_y) - exp(abc[0] * double(_x) * double(_x) + abc[1] * double(_x) + abc[2]); // y-exp(ax^2+bx+c)
        return true;
    }

    const double _x, _y; // x,y数据
};

int main()
{
    double ar = 1.0, br = 2.0, cr = 1.0;  // 真实参数值
    double ae = 2.0, be = -1.0, ce = 5.0; // 估计参数值
    int N = 100;                          // 数据点
    double w_sigma = 1.0;                 // 噪声Sigma值
    double inv_sigma = 1.0 / w_sigma;
    // cv::RNG rng; // OpenCV随机数产生器

    vector<double> x_data, y_data; // 数据
    for (int i = 0; i < N; i++)
    {
        double x = i / 100.0;
        x_data.push_back(x);
        // y_data.push_back(exp(ar * x * x + br * x + cr) + rng.gaussian(w_sigma * w_sigma));
        y_data.push_back(exp(ar * x * x + br * x + cr));
    }

    double abc[3] = {ae, be, ce};
    double *residual=new double;
    for (int i = 0; i < N; i++)
    {
        CURVE_FITTING_COST obj(x_data[i], y_data[i]);
        if(obj(abc,residual))cout<<residual[0]<<" ";
    }
    cout<<endl;
    delete residual;
    residual=nullptr;
}