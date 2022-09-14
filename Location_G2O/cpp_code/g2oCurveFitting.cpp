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
#include <string>

using namespace std;

// anchor position
vector<pair<double, double>> anchor{{0, 0}, {0, 172}, {93, 172}, {93, 0}, {46, 70}, {46, 125}};

// 优化点直接定义为Eigen::Vector2d,分别为x，y

// 定位模型的顶点，模板参数：优化变量维度和数据类型（x,y,theta,spd）
class myVertex : public g2o::BaseVertex<2, Eigen::Vector2d>
{
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  // 初始化
  virtual void setToOriginImpl() override
  {
    // 初始化theta为0默认初始向x正方向移动，需优化
    _estimate << 0, 0;
  }

  // 更新
  virtual void oplusImpl(const double *update) override
  {
    _estimate += Eigen::Vector2d(update);
  }

  // 存盘和读盘：留空
  virtual bool read(istream &in) {}

  virtual bool write(ostream &out) const {}
};

// 误差模型 模板参数：观测值维度，类型，连接顶点类型
// 观测值维度为6，则是将6个测距值算6条边；
typedef Eigen::Matrix<double, 6, 1> Vector6d;

class myEdge : public g2o::BaseUnaryEdge<6, Vector6d, myVertex>
{
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  // myEdge(vector<double> distance) : BaseUnaryEdge(), _dist(distance) {}
  // myEdge() : BaseUnaryEdge() {}

  // 计算定位模型误差
  virtual void computeError() override
  {
    // 获取该边所连顶点的指针
    const myVertex *v = reinterpret_cast<const myVertex *>(_vertices[0]);
    const Eigen::Vector2d particle = v->estimate();
    // e=sigam求和(|z2-((x-xanchor)2+(y-yanchor)2)|)
    // 误差
    _error(0, 0) = fabs(_measurement(0, 0) * _measurement(0, 0) - ((particle(0, 0) - anchor[0].first) * (particle(0, 0) - anchor[0].first) + (particle(1, 0) - anchor[0].second) * (particle(1, 0) - anchor[0].second)));
    _error(1, 0) = fabs(_measurement(1, 0) * _measurement(1, 0) - ((particle(0, 0) - anchor[1].first) * (particle(0, 0) - anchor[1].first) + (particle(1, 0) - anchor[1].second) * (particle(1, 0) - anchor[1].second)));
    _error(2, 0) = fabs(_measurement(2, 0) * _measurement(2, 0) - ((particle(0, 0) - anchor[2].first) * (particle(0, 0) - anchor[2].first) + (particle(1, 0) - anchor[2].second) * (particle(1, 0) - anchor[2].second)));
    _error(3, 0) = fabs(_measurement(3, 0) * _measurement(3, 0) - ((particle(0, 0) - anchor[3].first) * (particle(0, 0) - anchor[3].first) + (particle(1, 0) - anchor[3].second) * (particle(1, 0) - anchor[3].second)));
    _error(4, 0) = fabs(_measurement(4, 0) * _measurement(4, 0) - ((particle(0, 0) - anchor[4].first) * (particle(0, 0) - anchor[4].first) + (particle(1, 0) - anchor[4].second) * (particle(1, 0) - anchor[4].second)));
    _error(5, 0) = fabs(_measurement(5, 0) * _measurement(5, 0) - ((particle(0, 0) - anchor[5].first) * (particle(0, 0) - anchor[5].first) + (particle(1, 0) - anchor[5].second) * (particle(1, 0) - anchor[5].second)));
  }

  // // 计算雅可比矩阵
  // virtual void linearizeOplus() override
  // {
  //   // const CurveFittingVertex *v = static_cast<const CurveFittingVertex *>(_vertices[0]);
  //   // const Eigen::Vector3d abc = v->estimate();
  //   // double y = exp(abc[0] * _x * _x + abc[1] * _x + abc[2]);
  //   // _jacobianOplusXi[0] = -_x * _x * y;
  //   // _jacobianOplusXi[1] = -_x * y;
  //   // _jacobianOplusXi[2] = -y;
  // }

  virtual bool read(istream &in) {}

  virtual bool write(ostream &out) const {}

public:
  // vector<double> _dist; // dist 为_measurement
};

int main(int argc, char **argv)
{
  // 读取测量数据
  ifstream f;
  f.open("../../5-18-data/518data_1.txt");
  if (!f.is_open())
  {
    cout << "open file failed" << endl;
    f.close();
    return 0;
  }
  string s;
  vector<vector<double>> dist(300, vector<double>(6, 0));
  int idx = 0;
  while (getline(f, s))
  {
    // process data,读取距离数据
    // cout<<s<<endl;
    for (int jdx = 0; jdx < 6; jdx++)
    {
      dist[idx][jdx] = stod(s.substr(0 + jdx * 4, 3));
    }
    idx++;
  }
  f.close();
  int dist_size = idx;
  // for (int i = 0; i < idx; i++)
  // {
  //   for(int j=0;j<6;j++){
  //     cout<<dist[i][j]<<" ";
  //   }
  //   cout<<endl;
  // }
  double w_sigma = 1.0; // 噪声Sigma值
  double inv_sigma = 1.0 / w_sigma;
  // cv::RNG rng; // OpenCV随机数产生器

  // 构建图优化，先设定g2o
  typedef g2o::BlockSolver<g2o::BlockSolverTraits<2, 6>> BlockSolverType;           // 每个误差项优化变量维度为2(x,y)，误差值维度为6
  typedef g2o::LinearSolverDense<BlockSolverType::PoseMatrixType> LinearSolverType; // 线性求解器类型

  // 梯度下降方法，可以从GN, LM, DogLeg 中选
  auto solver = new g2o::OptimizationAlgorithmGaussNewton(
      g2o::make_unique<BlockSolverType>(g2o::make_unique<LinearSolverType>()));
  g2o::SparseOptimizer optimizer; // 图模型
  optimizer.setAlgorithm(solver); // 设置求解器
  optimizer.setVerbose(true);     // 打开调试输出

  vector<myVertex *> vs;
  // 根据dist数据行数增加顶点，滑动窗口如何删点？
  for (int i = 0; i < 3; i++)
  {
    // // 往图中增加顶点
    // CurveFittingVertex *v = new CurveFittingVertex();
    // v->setEstimate(Eigen::Vector3d(ae, be, ce));
    // v->setId(0);
    // optimizer.addVertex(v);
    myVertex *v = new myVertex();
    // 直接将所有点初始位置设为0,0，需优化
    v->setEstimate(Eigen::Vector2d(0, 0));
    v->setId(i);
    optimizer.addVertex(v);

    vs.emplace_back(v);

    // // 往图中增加边
    // for (int i = 0; i < 6; i++)
    // {
    //   CurveFittingEdge *edge = new CurveFittingEdge(x_data[i]);
    //   edge->setId(i);
    //   edge->setVertex(0, v);                                                                   // 设置连接的顶点
    //   edge->setMeasurement(y_data[i]);                                                         // 观测数值
    //   edge->setInformation(Eigen::Matrix<double, 1, 1>::Identity() * 1 / (w_sigma * w_sigma)); // 信息矩阵：协方差矩阵之逆
    //   optimizer.addEdge(edge);
    // }
    myEdge *edge = new myEdge();
    edge->setId(i);
    edge->setVertex(i, vs[i]);
    // 传入该时刻的6个测距值
    Vector6d vector_dist;
    vector_dist << dist[i][0], dist[i][1], dist[i][2], dist[i][3], dist[i][4], dist[i][5];
    // cout << "vector_dist: " << vector_dist.transpose() << endl;
    edge->setMeasurement(vector_dist);
    // 信息矩阵：协方差矩阵之逆 维数同误差维度相同
    edge->setInformation(Eigen::Matrix<double, 6, 6>::Identity() * 1 / (w_sigma * w_sigma));
    optimizer.addEdge(edge);
  }
  // 执行优化，滑动窗口还需要删点
  cout << "start optimization" << endl;
  chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
  optimizer.initializeOptimization();
  // 优化次数如何考虑？
  optimizer.optimize(10);
  chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
  chrono::duration<double> time_used = chrono::duration_cast<chrono::duration<double>>(t2 - t1);
  cout << "solve time cost = " << time_used.count() << " seconds. " << endl;

  // 输出优化值
  for (int i = 0; i < 3; i++)
  {
    Eigen::Vector2d p_estimate = vs[i]->estimate();
    cout << "estimated model: " << p_estimate.transpose() << endl;
  }

  return 0;
}