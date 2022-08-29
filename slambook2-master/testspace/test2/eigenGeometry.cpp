#include <iostream>
#include <cmath>
using namespace std;

#include <Eigen/Core>
#include <Eigen/Geometry>
using namespace Eigen;

int main(){
    //3D旋转矩阵直接使用Matrix3d或Matrix3f
    Matrix3d rotation_matrix = Matrix3d::Identity();
    // //旋转向量使用AngleAxis
    AngleAxisd rotation_vector(M_PI/4,Vector3d(0,0,1));//沿Z轴旋转45度
    // cout.precision(3);
    // // cout<<"rotation_matrix=\n"<<rotation_vector.matrix()<<endl;
    rotation_matrix = rotation_vector.toRotationMatrix();
    // //1/用AngleAxis进行坐标变换
    Vector3d v(1,0,0);
    Vector3d v_rotated=rotation_vector*v;
    // // cout<<"(1,0,0) after rotation (by angle axis)="<< v_rotated.transpose()<<endl;
    // //2/用旋转矩阵
    v_rotated=rotation_matrix*v;
    // // cout<<"(1,0,0) after rotation (by matrix)="<< v_rotated.transpose()<<endl;

    //欧拉角
    Vector3d euler_angles = rotation_matrix.eulerAngles(2,1,0); //ZYX顺序
    cout<<"ypr="<<euler_angles.transpose()<<endl;
    
    //欧式变换矩阵使用Eigen::Isometry
    Isometry3d T=Isometry3d::Identity();    //实质是4*4矩阵
    T.rotate(rotation_vector);  //按照rotation_vector进行旋转
    T.pretranslate(Vector3d(1,3,4));    //把平移向量设成（1，3，4）
    cout<<"Transform matrix=\n"<<T.matrix()<<endl;

    //用变换矩阵进行坐标变换
    Vector3d v_transformed = T*v;   //相当于R*v+t
    cout<<"v transformed = "<<v_transformed.transpose()<<endl;

    //四元数
    Quaterniond q = Quaterniond(rotation_vector);
    cout<<"quaternion from rotarion vector="<<q.coeffs().transpose()<<endl;//coeffs的顺序是（x,y,z,w)w为实部

    return 0;
}