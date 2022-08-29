#include <iostream>
using namespace std;

#include <ctime>

//Eigen核心
#include <Eigen/Core>
//稠密矩阵的代数运算
#include <Eigen/Dense>
using namespace Eigen;

#define MATRIX_SIZE 50

//Eigen基本类型的使用
int main(){
    //Eigen中所有向量和矩阵都是Eigen::Matrix，它是一个模板类。前三个参数为数据类型,行,列
    //声明一个2*3的float矩阵
    Matrix<float,2,3>matrix_23;

    //同时，Eigen通过typedef提供许多内置类型，不过底层仍是Eigen::Matrix
    //例如，Vector3d实质上是Eigen::Matrix<double,3,1>
    
    Vector3d v_3d;
    Matrix<float,3,1>vd_3d;
    //v_3d==vd_3d

    //Matrix3d实质上是Eigen::Matrix<double,3,3>
    Matrix3d matrix_33=Matrix3d::Zero();//初始化为零
    
    //如果不确定矩阵大小，可以使用动态大小的矩阵
    Matrix<double,Dynamic,Dynamic>matrix_dynamic;
    MatrixXd matrix_x;
    //matrix_dynamic==matric_x

    //Eigen矩阵操作
    //输入数据（初始化）
    matrix_23<<1,2,3,4,5,6;
    //输出
    // cout<<"matirx 2x3:\n"<<matrix_23<<endl;

    // //用（）访问矩阵中的元素
    // cout<<"print matrix 2x3: "<<endl;
    // for(int i=0;i<2;i++){
    //     for(int j=0;j<3;j++)cout<<matrix_23(i,j)<<"\t";
    //     cout<<endl;
    // }

    //矩阵向量相乘
    v_3d<<3,2,1;
    vd_3d<<4,5,6;

    //Eigen不能混合两种不同类型的矩阵,也不能弄错矩阵的维度
    // Matrix<double,2,1>result_wrong_type = matrix_23*v_3d;
    // Matrix<double,2,3>result_wrong_dimension=matrix_23.cast<double>()*v_3d;
    //显式转换
    // Matrix<double,2,1>result=matrix_23.cast<double>()*v_3d;
    // cout << "[1,2,3;4,5,6]*[3,2,1]=" << result.transpose() << endl;

    //矩阵的迹
    //matrix_33.trace();
    //矩阵的逆
    //matrix_33.inverse();
    //矩阵的行列是
    //matrix_33.determinant()

    //特征值
    //实对称矩阵可以保证对角化成功
    matrix_33=Matrix3d::Random();   //随机数矩阵
    //埃尔米特矩阵等于自己的共轭转置。根据有限维的谱定理必定存在着一个正交归一基，可以表达自伴算子为一个实值的对角矩阵。
    SelfAdjointEigenSolver<Matrix3d> eigen_solver(matrix_33.transpose()*matrix_33);
    // cout<<"Eigen valuse= \n"<<eigen_solver.eigenvalues()<<endl;
    // cout<<"Eigen vectors= \n"<<eigen_solver.eigenvectors()<<endl;

    //解方程
    //求解matirx_NN*x=v_Nd方程
    Matrix<double,MATRIX_SIZE,MATRIX_SIZE>matrix_NN
        =MatrixXd::Random(MATRIX_SIZE,MATRIX_SIZE);
        matrix_NN=matrix_NN*matrix_NN.transpose();  //确保半正定
        Matrix<double,MATRIX_SIZE,1>v_Nd=MatrixXd::Random(MATRIX_SIZE,1);

        //1,直接求逆，运算量大
        clock_t time_stt=clock();
        Matrix<double,MATRIX_SIZE,1>x=matrix_NN.inverse()*v_Nd;
        cout<<"time of normal inverse is"
            <<1000*(clock()-time_stt)/(double)CLOCKS_PER_SEC<<"ms"<<endl;
        cout<<"x="<<x.transpose()<<endl;

        //2,矩阵分解求解，如QR
        time_stt=clock();
        x=matrix_NN.colPivHouseholderQr().solve(v_Nd);
        cout<<"time of Qr decompositon is"
            <<1000*(clock()-time_stt)/(double)CLOCKS_PER_SEC<<"ms"<<endl;
        cout<<"x="<<x.transpose()<<endl;



        return 0;
}