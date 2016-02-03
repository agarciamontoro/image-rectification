#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "util.hpp"

#include <iostream>

using namespace cv;
using namespace std;

int main(){
    /****************** EPIPOLAR GEOMETRY **************************/
    // // Mat img_1 = imread("img/Vmort1.pgm");
    // // Mat img_2 = imread("img/Vmort2.pgm");

    Mat img_1 = imread("img/monitogo.png");
    Mat img_2 = imread("img/monitogo2.png");

    Mat fund_mat;

    Vec3d epipole1, epipole2;
    computeAndDrawEpiLines(img_1, img_2, 150, epipole1, epipole2, fund_mat);

    Mat A, B, Ap, Bp;

    Mat e_x = crossProductMatrix(epipole1);

    /****************** PROJECTIVE **************************/

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

    Mat Hp_p = Mat::eye(3, 3, CV_64F);
    Hp_p.at<double>(2,0) = wp.at<double>(0,0);
    Hp_p.at<double>(2,1) = wp.at<double>(0,1);

    /****************** SIMILARITY **************************/

    double vp_c = getMinimumYcoordinate(img_1, img_2, H_p, Hp_p);

    /****************** RECTIFY IMAGES **********************/

    cout << "H_p = " << H_p << endl;
    cout << "Hp_p = " << Hp_p << endl;

    Mat img_1_dst = Mat::zeros(512,512,CV_64F);
    Mat img_2_dst = Mat::zeros(512,512,CV_64F);

    warpPerspective( img_1, img_1_dst, H_p, img_1.size() );
    warpPerspective( img_2, img_2_dst, Hp_p, img_2.size() );

    draw(img_1, "1");
    draw(img_1_dst, "1 proyectada");

    draw(img_2, "2");
    draw(img_2_dst, "2 proyectada");

    waitKey();

    destroyAllWindows();
}
