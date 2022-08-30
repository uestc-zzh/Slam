#include <iostream>
#include <fstream>
#include <opencv2/opencv.hpp>
#include <Eigen/Core>
#include <Eigen/Dense>

using namespace std;
using namespace Eigen;

int main(int argc, char **argv)
{
    double ar = 1.0, br = 2.0, cr = 1.0;  // 真实参数值
    double ae = 2.0, be = -1.0, ce = 5.0; // 估计参数值
    int N = 100;                          // 数据点
    double w_sigma = 1.0;                 // 噪声sigma值
    double inv_sigma = 1.0 / w_sigma;
    cv::RNG rng; // opencv随机数产生器

    ofstream output;
    output.open("../origin_data.txt");
    vector<double> x_data, y_data; // 数据
    for (int i = 0; i < N; i++)
    {
        double x = i / 100.0;
        x_data.push_back(x);
        y_data.push_back(exp(ar * x * x + br * x + cr) + rng.gaussian(w_sigma * w_sigma));
        output<<y_data[i]<<endl;
    }
    output.close();

    // 开始Gauss-Newton迭代
    int iterations = 100;
    double cost = 0, lastCost = 0; // 本次迭代的cost和上一次迭代的cost

    // chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
    for (int iter = 0; iter < iterations; iter++)
    {

        Matrix3d H = Matrix3d::Zero(); // Hessian = J^T W^{-1} J in Gauss-Newton
        Vector3d b = Vector3d::Zero(); // bias
        cost = 0;

        for (int i = 0; i < N; i++)
        {
            double xi = x_data[i], yi = y_data[i]; // data[i]
            double error = yi - exp(ae * xi * xi + be * xi + ce);
            Vector3d J;                                         // Jacobian Array
            J[0] = -xi * xi * exp(ae * xi * xi + be * xi + ce); // de/da
            J[1] = -xi * exp(ae * xi * xi + be * xi + ce);      // de/db
            J[2] = -exp(ae * xi * xi + be * xi + ce);           // de/dc

            H += inv_sigma * inv_sigma * J * J.transpose();
            b += -inv_sigma * inv_sigma * error * J;

            cost += error * error;
        }

        // 求解线性方程Hx=b
        Vector3d dx = H.ldlt().solve(b);
        if (isnan(dx[0]))
        {
            cout << "result is nan!" << endl;
            break;
        }

        if (iter > 0 && cost >= lastCost)
        {
            cout << "cost: " << cost << ">= last cost: " << lastCost << ", break." << endl;
            break;
        }

        ae += dx[0];    // 将自变量增加相应步长，继续下一轮的迭代
        be += dx[1];
        ce += dx[2];

        lastCost = cost;

        cout << "total cost: " << cost << ", \t\tupdate: " << dx.transpose() << "\t\testimated params: " << ae << "," << be << "," << ce << endl;
    }

    // chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
    // chrono::duration<double> time_used = chrono::duration_cast<chrono::duration<double>>(t2 - t1);
    // cout << "solve time cost = " << time_used.count() << " seconds. " << endl;
    // cout << "estimated abc = " << ae << ", " << be << ", " << ce << endl;
    output.open("../out_data.txt");
    vector<double>out_data;
    for(int i=0;i<N;i++){
        double x=i/100.0;
        out_data.push_back(exp(ae*x*x+be*x+ce));
        output<<out_data[i]<<endl;
    }
    output.close();

    return 0;
}