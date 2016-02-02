#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "util.hpp"

#include <iostream>

using namespace cv;
using namespace std;

int main(){
    Mat img_1 = imread("img/Vmort1.pgm");
    Mat img_2 = imread("img/Vmort2.pgm");

    int width = img_1.cols;
    int height = img_1.rows;

    int size = 3;

    Mat PPt = Mat::zeros(size, size, CV_64F);

    PPt.at<double>(0,0) = width*width - 1;
    PPt.at<double>(1,1) = height*height - 1;

    PPt *= (width*height) / 12;

    double w_1 = width - 1;
    double h_1 = height - 1;

    double values[3][3] = {
        {w_1*w_1, w_1*h_1, 2*w_1},
        {w_1*h_1, h_1*h_1, 2*h_1},
        {2*w_1, 2*h_1, 4}
    };

    Mat pcpct(size, size, CV_64F, values);

    pcpct /= 4;

    cout << PPt << endl << pcpct << endl;

    Vec3d epipole1, epipole2;
    computeAndDrawEpiLines(img_1, img_2, 150, epipole1, epipole2);

    cout << epipole1 << endl << epipole2 << endl;

    draw(img_1, "1");
    draw(img_2, "2");

    waitKey();

    destroyAllWindows();

}
