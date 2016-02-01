#include <opencv2/core/core.hpp>
#include <iostream>

using namespace cv;
using namespace std;

// https://github.com/Itseez/opencv/blob/master/modules/stitching/src/autocalib.cpp
bool choleskyDecomp(Mat A, Mat &D){
    size_t astep = A.step;
    double* data = A.ptr<double>();
    int size = A.cols;

    if (hal::Cholesky(data, astep, size, 0, 0, 0))
    {
        astep /= sizeof(data[0]);
        for (int i = 0; i < size; ++i){
            data[i*astep + i] = (double)(1./data[i*astep + i]);
        }

        D = A.clone();
        D.at<double>(0,1) = D.at<double>(0,2) = D.at<double>(1,2) = 0;

        D = D.t();

        return true;
    }

    return false;
}

int main(){
    int size = 3;

    double vals[3][3] = {
        {4, 12, -16},
        {12, 37, -43},
        {-16, -43, 98}
    };

    Mat A = Mat(size, size, CV_64F, vals);
    Mat D;

    cout << A << endl;

    if(choleskyDecomp(A,D)){
        cout << D << endl << D.t()*D << endl;
    }
}
