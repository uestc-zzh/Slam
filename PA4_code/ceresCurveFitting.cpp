#include <iostream>
#include <fstream>
#include <opencv2/core/core.hpp>
#include <ceres/ceres.h>
#include <chrono>

using namespace std;

// cost function model
struct CURVE_FITTING_COST
{
    CURVE_FITTING_COST(double x, double y) : _x(x), _y(y) {}

    // 残差计算
    // 重载operator()的对象，也称Functor, 作为可调用对象。可调用对象包括了函数指针、重载operator()的对象以及可隐式转化为前两者的对象
    template <typename T>
    bool operator()(
        const T *const abc, // 模型参数,有3维
        T *residual) const
    {
        residual[0] = T(_y) - ceres::exp(abc[0] * T(_x) * T(_x) + abc[1] * T(_x) + abc[2]); // y-exp(ax^2+bx+c)
        return true;
    }

    const double _x, _y; // x,y数据
};

int main(int argc, char **argv)
{
    double ar = 1.0, br = 2.0, cr = 1.0;  // real parameters
    double ae = 2.0, be = -1.0, ce = 6.0; // estimated parameters
    int N = 100;
    double w_sigma = 1.0;
    double inv_sigma = 1.0 / w_sigma;
    cv::RNG rng; // OpenCV随机数产生器

    fstream output;
    output.open("../ceresData/origin_data.txt");
    vector<double> x_data, y_data;
    for (int i = 0; i < N; i++)
    {
        double x = i / 100.0;
        x_data.push_back(x);
        y_data.push_back(exp(ar * x * x + br * x + cr) + rng.gaussian(w_sigma * w_sigma));
        output << y_data[i] << endl;
    }
    output.close();

    double abc[3] = {ae, be, ce};

    // construct least sqaure problem
    ceres::Problem problem;
    for (int i = 0; i < N; i++)
    {
        problem.AddResidualBlock( // 向问题中添加误差项
                                  // 使用自动求带，模板参数：误差类型，输出维度，输入维度，维数要与前面struct中一致
            new ceres::AutoDiffCostFunction<CURVE_FITTING_COST, 1, 3>(
                new CURVE_FITTING_COST(x_data[i], y_data[i])),
            nullptr, // 核函数，这里不使用，为空
            abc      // 待估计参数
        );
    }

    // 配置求解器
    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_NORMAL_CHOLESKY; // 增量方程如何求解
    options.minimizer_progress_to_stdout = true;               // 输出到cout

    ceres::Solver::Summary summary; // 优化信息
    chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
    ceres::Solve(options, &problem, &summary); //开始优化
    chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
    chrono::duration<double> time_used = chrono::duration_cast<chrono::duration<double>>(t2 - t1);
    cout << "solve time cost = " << time_used.count() << " seconds." << endl;

    // 输出结果
    cout << summary.BriefReport() << endl;
    cout << "estimated a,b,c = ";
    output.open("../ceresData/fitting_paras.txt");
    for (auto para : abc)
    {
        output << para << " ";
        cout << para << " ";
    }
    output.close();
    cout << endl;

    return 0;
}