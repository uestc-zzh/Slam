#include <iostream>
#include <vector>
#include <algorithm>

using namespace std;

#include <Eigen/Core>
#include <Eigen/Geometry>

using namespace Eigen;

int main(int argc,char **argv){
    Quaterniond q1(0.55, 0.3, 0.2, 0.2),q2(-0.1, 0.3, -0.7, 0.2);
    q1.normalize();
    q2.normalize();
    Vector3d t1(0.7,1.1,0.2),t2(-0.1,0.4,0.8);
    Vector3d p1(0.5,-0.1,0.2);

    //初始化为单位阵
    // Isometry3d T1=Isometry3d::Identity();
    // Isometry3d T2=Isometry3d::Identity();
    // T1.rotate(q1);
    // T2.rotate(q2);
    Isometry3d T1(q1),T2(q2);
    T1.pretranslate(t1);
    T2.pretranslate(t2);

    Vector3d p2=T2*T1.inverse()*p1;
    cout<<p2.transpose()<<endl;
    return 0;
}