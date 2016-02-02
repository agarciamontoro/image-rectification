#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "util.hpp"

#include <iostream>

using namespace cv;
using namespace std;

int main(){
    Mat img_1 = imread("img/Vmort1.pgm");
    Mat img_2 = imread("img/Vmort2.pgm");

    cout << img_1.cols << " " << img_1.rows << endl;

    Mat fund_mat;

    Vec3d epipole1, epipole2;
    computeAndDrawEpiLines(img_1, img_2, 150, epipole1, epipole2, fund_mat);

    Mat A, B, Ap, Bp;

    Mat e_x = crossProductMatrix(epipole1);

    obtainAB(img_1, e_x, A, B);
    obtainAB(img_2, fund_mat, Ap, Bp);

    cout << "A = " << A << endl;
    cout << "B = " << B << endl;
    Vec3d z = getInitialGuess(A, B, Ap, Bp);

    cout << "z = " << z << endl;

    Mat w = e_x * Mat(z);
    Mat wp = fund_mat * Mat(z);

    Mat H_p = Mat::eye(3, 3, CV_64F);

    H_p.at<double>(2,0) = w.at<double>(0,0);
    H_p.at<double>(2,1) = w.at<double>(0,1);
    cout << "H_p = " << H_p << endl;

    draw(img_1, "1");

    Mat img_1_dst = Mat::zeros(1024,1024,CV_64F);
    warpPerspective( img_1, img_1_dst, H_p, Size(1024,1024) );

    draw(img_1_dst, "AKLSJDLKAJSDLASD");

    waitKey();

    destroyAllWindows();

}
