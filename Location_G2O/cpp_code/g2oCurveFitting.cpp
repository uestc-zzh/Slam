#include <iostream>
#include <fstream>
#include <g2o/core/g2o_core_api.h>
#include <g2o/core/base_vertex.h>
#include <g2o/core/base_unary_edge.h>
#include "g2o/core/base_binary_edge.h"
#include <g2o/core/block_solver.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/core/optimization_algorithm_gauss_newton.h>
#include <g2o/core/optimization_algorithm_dogleg.h>
#include <g2o/solvers/dense/linear_solver_dense.h>
#include <g2o/core/robust_kernel_impl.h>
#include <Eigen/Core>
#include <opencv2/core/core.hpp>
#include <cmath>
#include <chrono>
#include <vector>
#include <utility>
#include <string>

using namespace std;

#define MAX_DIST_SIZE 1000
#define WINDOW_SIZE 10

// anchor position
vector<pair<double, double>> anchor{{-0.72, 2.0}, {-0.65, -2.0}, {0.65, -2.0}, {0.72, 2.0}, {0.0, 1.35}, {0.0, -1.58}};

// 优化点直接定义为Eigen::Vector2d,分别为x，y

// 定位模型的顶点，模板参数：优化变量维度和数据类型（x,y,theta,spd）
class myVertex : public g2o::BaseVertex<2, Eigen::Vector2d>
{
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  // 初始化
  virtual void setToOriginImpl() override
  {
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

  // 计算定位模型误差
  virtual void computeError() override
  {
    // 获取该边所连顶点的指针
    const myVertex *v = reinterpret_cast<const myVertex *>(_vertices[0]);
    const Eigen::Vector2d particle = v->estimate();

    // 误差
    _error(0, 0) = fabs(_measurement(0, 0) * _measurement(0, 0) - ((particle(0, 0) - anchor[0].first) * (particle(0, 0) - anchor[0].first) + (particle(1, 0) - anchor[0].second) * (particle(1, 0) - anchor[0].second)));
    _error(1, 0) = fabs(_measurement(1, 0) * _measurement(1, 0) - ((particle(0, 0) - anchor[1].first) * (particle(0, 0) - anchor[1].first) + (particle(1, 0) - anchor[1].second) * (particle(1, 0) - anchor[1].second)));
    _error(2, 0) = fabs(_measurement(2, 0) * _measurement(2, 0) - ((particle(0, 0) - anchor[2].first) * (particle(0, 0) - anchor[2].first) + (particle(1, 0) - anchor[2].second) * (particle(1, 0) - anchor[2].second)));
    _error(3, 0) = fabs(_measurement(3, 0) * _measurement(3, 0) - ((particle(0, 0) - anchor[3].first) * (particle(0, 0) - anchor[3].first) + (particle(1, 0) - anchor[3].second) * (particle(1, 0) - anchor[3].second)));
    _error(4, 0) = fabs(_measurement(4, 0) * _measurement(4, 0) - ((particle(0, 0) - anchor[4].first) * (particle(0, 0) - anchor[4].first) + (particle(1, 0) - anchor[4].second) * (particle(1, 0) - anchor[4].second)));
    _error(5, 0) = fabs(_measurement(5, 0) * _measurement(5, 0) - ((particle(0, 0) - anchor[5].first) * (particle(0, 0) - anchor[5].first) + (particle(1, 0) - anchor[5].second) * (particle(1, 0) - anchor[5].second)));
  }

  virtual bool read(istream &in) {}

  virtual bool write(ostream &out) const {}
};

class constraitEdge : public g2o::BaseBinaryEdge<1, double, myVertex, myVertex>
{
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  // 计算定位模型误差
  virtual void computeError() override
  {
    // 获取该边所连顶点的指针
    const myVertex *v1 = reinterpret_cast<const myVertex *>(_vertices[0]);
    const myVertex *v2 = reinterpret_cast<const myVertex *>(_vertices[1]);
    const Eigen::Vector2d particle1 = v1->estimate();
    const Eigen::Vector2d particle2 = v2->estimate();

    _error(0, 0) = _measurement - (particle1 - particle2).norm();
  }

  virtual bool read(istream &in) {}

  virtual bool write(ostream &out) const {}
};

int main(int argc, char **argv)
{
  // 读取测量数据
  ifstream f;
  f.open("../../5-18-data/dingdian14.txt");
  if (!f.is_open())
  {
    cout << "open file failed" << endl;
    f.close();
    return 0;
  }
  string s;
  vector<vector<double>> dist(MAX_DIST_SIZE, vector<double>(6, 0));
  int idx = 0;
  // while (getline(f, s))
  // {
  //   for (int jdx = 0; jdx < 6; jdx++)
  //   {
  //     dist[idx][jdx] = stod(s.substr(0 + jdx * 4, 3));
  //     cout<<dist[idx][jdx]<<" ";
  //   }
  //   idx++;
  // }
  while (getline(f, s)) {
    std::istringstream ss(s);
    for (int jdx = 0; jdx < 6; jdx++) {
      ss >> dist[idx][jdx];
    }
    idx++;
  }
  f.close();

  int dist_size = idx;

  // 输出优化值
  ofstream outputFile("g2o.txt");

  // Check if the file is successfully opened
  if (!outputFile.is_open()) {
    std::cerr << "open file FAILED." << std::endl;
    return 0;
  }

  // 构建图优化，先设定g2o
  typedef g2o::BlockSolver<g2o::BlockSolverTraits<2, 1>> BlockSolverType;           // 每个误差项优化变量维度为2(x,y)，误差值维度为6
  typedef g2o::LinearSolverDense<BlockSolverType::PoseMatrixType> LinearSolverType; // 线性求解器类型

  // 梯度下降方法，可以从GN, LM, DogLeg 中选
  auto solver = new g2o::OptimizationAlgorithmGaussNewton(
      g2o::make_unique<BlockSolverType>(g2o::make_unique<LinearSolverType>()));
  g2o::SparseOptimizer optimizer; // 图模型
  optimizer.setAlgorithm(solver); // 设置求解器
  optimizer.setVerbose(true);     // 打开调试输出

  double p_sigma = 2.0; // 预测噪声Sigma值
  double w_sigma = 1.0; // 测量噪声Sigma值
  vector<myVertex *> vs;
  vector<constraitEdge *> c_edges;
  vector<myEdge *> m_edges;

  for (int i = 0; i < dist_size; i++)
  {

    myVertex *v = new myVertex();

    if (i == 0)
    {
      v->setEstimate(Eigen::Vector2d(0, 0));
    }
    else
    {
      v->setEstimate(Eigen::Vector2d(vs[i - 1]->estimate()(0), vs[i - 1]->estimate()(1)));
    }
    v->setId(i);
    optimizer.addVertex(v);

    vs.emplace_back(v);

    if (i > 0)
    {
      constraitEdge *e = new constraitEdge();
      e->setId(i + dist_size);
      e->setVertex(0, vs[i]);
      e->setVertex(1, vs[i - 1]);
      e->setMeasurement(0.1); // 约束相邻点之间的距离
      e->setInformation(Eigen::Matrix<double, 1, 1>::Identity()* 1 / (p_sigma * p_sigma)); // 设置信息矩阵
      e->setRobustKernel(new g2o::RobustKernelHuber());
      optimizer.addEdge(e);

      c_edges.emplace_back(e);
    }

  // 传入该时刻的6个测距值
    Vector6d vector_dist;
    vector_dist << dist[i][0], dist[i][1], dist[i][2], dist[i][3], dist[i][4], dist[i][5];
    
    myEdge *edge = new myEdge();
    edge->setId(i);
    edge->setVertex(0, vs[i]);
    edge->setMeasurement(vector_dist);
    // 信息矩阵：协方差矩阵之逆 维数同误差维度相同
    edge->setInformation(Eigen::Matrix<double, 6, 6>::Identity() * 1 / (w_sigma * w_sigma));
    edge->setRobustKernel(new g2o::RobustKernelHuber());
    optimizer.addEdge(edge);

    m_edges.emplace_back(edge);

    if (i >= WINDOW_SIZE)
    {
      // 删除窗口中最早的顶点及其相连的边
      myVertex *old_vertex = vs[i - WINDOW_SIZE];
      optimizer.removeVertex(old_vertex);

      constraitEdge *old_c_edge = c_edges[i - WINDOW_SIZE];
      optimizer.removeEdge(old_c_edge);

      myEdge *old_m_edge = m_edges[i - WINDOW_SIZE];
      optimizer.removeEdge(old_m_edge); 

      vs[i-WINDOW_SIZE]->setFixed(true);
      optimizer.initializeOptimization();
      optimizer.optimize(5);
      outputFile << vs[i-WINDOW_SIZE]->estimate().transpose() << endl;
    }
    else {
      optimizer.initializeOptimization();
      optimizer.optimize(5);
      outputFile << vs[i]->estimate().transpose() << endl;
    }

    // cout << "Vertex " << vs[0]->id() << ": " << vs[0]->estimate().transpose() << endl;
  }

  // cout << "start optimization" << endl;
  // chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
  // optimizer.initializeOptimization();
  // optimizer.optimize(10);
  // chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
  // chrono::duration<double> time_used = chrono::duration_cast<chrono::duration<double>>(t2 - t1);
  // cout << "solve time cost = " << time_used.count() << " seconds. " << endl;

  
  // for (int i = 0; i < dist_size; i++)
  // {
  //   Eigen::Vector2d p_estimate = vs[i]->estimate();
  //   // cout << p_estimate.transpose() << endl;
  //   outputFile << p_estimate.transpose() << endl;
  // }
  
  // for (const auto& vertex : vs) {
  //     cout << "Vertex " << vertex->id() << ": " << vertex->estimate().transpose() << endl;
  // }

  return 0;
}