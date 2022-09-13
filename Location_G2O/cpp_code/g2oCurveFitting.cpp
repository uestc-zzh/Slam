#include <iostream>
#include <fstream>
#include <g2o/core/g2o_core_api.h>
#include <g2o/core/base_vertex.h>
#include <g2o/core/base_unary_edge.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/core/optimization_algorithm_gauss_newton.h>
#include <g2o/core/optimization_algorithm_dogleg.h>
#include <g2o/solvers/dense/linear_solver_dense.h>
#include <Eigen/Core>
#include <opencv2/core/core.hpp>
#include <cmath>
#include <chrono>
#include <vector>
#include <utility>

#define PI acos(-1)

using namespace std;

// anchor position
vector<pair<double, double>> anchor{{0, 0}, {0, 1.72}, {0.93, 1.72}, {0.93, 0}, {0.46, 0.70}, {0.46, 1.25}};

// 优化点属性
class Particles
{
  Particles() : x(0), y(0), theta(0), spd(1.0) {}
  Particles(double x, double y, double theta, double speed) : x(x), y(y), theta(theta), spd(speed) {}
  double x;
  double y;
  double theta;
  double spd;
};

// 定位模型的顶点，模板参数：优化变量维度和数据类型（x,y,theta,spd）
class myVertex : public g2o::BaseVertex<4, Particles>
{
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  myVertex() {}
  // 初始化
  virtual void setToOriginImpl() override
  {
    _estimate = Particles();
  }

  // 更新
  virtual void oplusImpl(const double *update) override
  {
    update = reinterpret_cast<const Particles *>(update);
    cv::RNG rng;
    _estimate.x = update.x + rng.gaussian(1.0 * 1.0);
    _estimate.y = update.y + rng.gaussian(1.0 * 1.0);
    _estimate.theta = fmod(update.theta, 2 * PI); // 取模
    _estimate.spd = update.spd + rng.gaussian(1.0 * 1.0);
    if (_estimate.spd >= 5 || _estimate.spd < 0)
    {
      _estimate.spd = 1.0;
    }
    double dist = (_estimate.spd * 1.0) + rng.gaussian(1.0 * 1.0);
    _estimate.x += cos(_estimate.theta) * dist;
    _estimate.y += sin(_estimate.theta) * dist;
  }

  // 存盘和读盘：留空
  virtual bool read(istream &in) {}

  virtual bool write(ostream &out) const {}
};

// 误差模型 模板参数：观测值维度，类型，连接顶点类型
// 观测值维度为6？ 观测值类型为6个测距值的vector?
class myEdge : public g2o::BaseUnaryEdge<6, vector<double>distance, myVertex>
{
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  // myEdge(vector<double>distance) : BaseUnaryEdge(), _dist(distance) {}
  myEdge() : BaseUnaryEdge(){}

  // 计算定位模型误差
  virtual void computeError() override
  {
    const myVertex *v = reinterpret_cast<const Particles *>(_vertices[0]);
    const Particles particle = v->estimate();
    // e=sigam求和(|z2-((x-xanchor)2+(y-yanchor)2)|)
    // x的误差
    _error(0, 0) = fabs(_measurement[0] * _measurement[0] - ((particle.x - anchor[0].first) * (particle.x - anchor[0].first) + (particle.y - anchor[0].second) * (particle.y - anchor[0].second)))*cos(particle.theta) +
                   fabs(_measurement[1] * _measurement[1] - ((particle.x - anchor[1].first) * (particle.x - anchor[1].first) + (particle.y - anchor[1].second) * (particle.y - anchor[1].second)))*cos(particle.theta) +
                   fabs(_measurement[2] * _measurement[2] - ((particle.x - anchor[2].first) * (particle.x - anchor[2].first) + (particle.y - anchor[2].second) * (particle.y - anchor[2].second)))*cos(particle.theta) +
                   fabs(_measurement[3] * _measurement[3] - ((particle.x - anchor[3].first) * (particle.x - anchor[3].first) + (particle.y - anchor[3].second) * (particle.y - anchor[3].second)))*cos(particle.theta) +
                   fabs(_measurement[4] * _measurement[4] - ((particle.x - anchor[4].first) * (particle.x - anchor[4].first) + (particle.y - anchor[4].second) * (particle.y - anchor[4].second)))*cos(particle.theta) +
                   fabs(_measurement[5] * _measurement[5] - ((particle.x - anchor[5].first) * (particle.x - anchor[5].first) + (particle.y - anchor[5].second) * (particle.y - anchor[5].second)))*cos(particle.theta);
    // y的误差
    _error(1, 0) = fabs(_measurement[0] * _measurement[0] - ((particle.x - anchor[0].first) * (particle.x - anchor[0].first) + (particle.y - anchor[0].second) * (particle.y - anchor[0].second)))*sin(particle.theta) +
                   fabs(_measurement[1] * _measurement[1] - ((particle.x - anchor[1].first) * (particle.x - anchor[1].first) + (particle.y - anchor[1].second) * (particle.y - anchor[1].second)))*sin(particle.theta) +
                   fabs(_measurement[2] * _measurement[2] - ((particle.x - anchor[2].first) * (particle.x - anchor[2].first) + (particle.y - anchor[2].second) * (particle.y - anchor[2].second)))*sin(particle.theta) +
                   fabs(_measurement[3] * _measurement[3] - ((particle.x - anchor[3].first) * (particle.x - anchor[3].first) + (particle.y - anchor[3].second) * (particle.y - anchor[3].second)))*sin(particle.theta) +
                   fabs(_measurement[4] * _measurement[4] - ((particle.x - anchor[4].first) * (particle.x - anchor[4].first) + (particle.y - anchor[4].second) * (particle.y - anchor[4].second)))*sin(particle.theta) +
                   fabs(_measurement[5] * _measurement[5] - ((particle.x - anchor[5].first) * (particle.x - anchor[5].first) + (particle.y - anchor[5].second) * (particle.y - anchor[5].second)))*sin(particle.theta);
    // theta的误差，暂时不考虑
    _error(2, 0) = 0.5;
    // spd的误差，暂时不考虑
    _error(3, 0) = 0.5;
  }

  // 计算雅可比矩阵
  virtual void linearizeOplus() override
  {
    // const CurveFittingVertex *v = static_cast<const CurveFittingVertex *>(_vertices[0]);
    // const Eigen::Vector3d abc = v->estimate();
    // double y = exp(abc[0] * _x * _x + abc[1] * _x + abc[2]);
    // _jacobianOplusXi[0] = -_x * _x * y;
    // _jacobianOplusXi[1] = -_x * y;
    // _jacobianOplusXi[2] = -y;
  }

  virtual bool read(istream &in) {}

  virtual bool write(ostream &out) const {}

public:
  vector<double> _dist; // dist 为_measurement
};

int main(int argc, char **argv)
{
  // 读取测量数据
  ifstream f;
  f.open("../5-18/data/outcar/1-1");
  if(!file.is_open()){
    cout<<"open file failed"<<endl;
    file.close();
    return 0;
  }
  string s;
  while(getline(file,s)){
    cout<<s<<endl;
  }
  file.close();
  
  // double w_sigma = 1.0;                 // 噪声Sigma值
  // double inv_sigma = 1.0 / w_sigma;
  // cv::RNG rng; // OpenCV随机数产生器

  // vector<double> x_data, y_data; // 数据
  // for (int i = 0; i < N; i++)
  // {
  //   double x = i / 100.0;
  //   x_data.push_back(x);
  //   y_data.push_back(exp(ar * x * x + br * x + cr) + rng.gaussian(w_sigma * w_sigma));
  // }

  // // 构建图优化，先设定g2o
  // typedef g2o::BlockSolver<g2o::BlockSolverTraits<3, 1>> BlockSolverType;           // 每个误差项优化变量维度为3，误差值维度为1
  // typedef g2o::LinearSolverDense<BlockSolverType::PoseMatrixType> LinearSolverType; // 线性求解器类型

  // // 梯度下降方法，可以从GN, LM, DogLeg 中选
  // auto solver = new g2o::OptimizationAlgorithmGaussNewton(
  //     g2o::make_unique<BlockSolverType>(g2o::make_unique<LinearSolverType>()));
  // g2o::SparseOptimizer optimizer; // 图模型
  // optimizer.setAlgorithm(solver); // 设置求解器
  // optimizer.setVerbose(true);     // 打开调试输出

  // // 往图中增加顶点
  // // CurveFittingVertex *v = new CurveFittingVertex();
  // // v->setEstimate(Eigen::Vector3d(ae, be, ce));
  // // v->setId(0);
  // // optimizer.addVertex(v);
  // myVertex *v = new myVertex();

  // // 往图中增加边
  // for (int i = 0; i < N; i++)
  // {
  //   CurveFittingEdge *edge = new CurveFittingEdge(x_data[i]);
  //   edge->setId(i);
  //   edge->setVertex(0, v);                                                                   // 设置连接的顶点
  //   edge->setMeasurement(y_data[i]);                                                         // 观测数值
  //   edge->setInformation(Eigen::Matrix<double, 1, 1>::Identity() * 1 / (w_sigma * w_sigma)); // 信息矩阵：协方差矩阵之逆
  //   optimizer.addEdge(edge);
  // }

  // // 执行优化
  // cout << "start optimization" << endl;
  // chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
  // optimizer.initializeOptimization();
  // optimizer.optimize(10);
  // chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
  // chrono::duration<double> time_used = chrono::duration_cast<chrono::duration<double>>(t2 - t1);
  // cout << "solve time cost = " << time_used.count() << " seconds. " << endl;

  // // 输出优化值
  // Eigen::Vector3d abc_estimate = v->estimate();
  // cout << "estimated model: " << abc_estimate.transpose() << endl;

  return 0;
}