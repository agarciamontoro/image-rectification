#ifndef __UTIL_HPP__
#define __UTIL_HPP__

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/calib3d/calib3d.hpp>

#include <iostream>

using namespace cv;
using namespace std;

#define INF 2147483647.0

enum detector_id{
    ORB,
    BRISK
};

enum descriptor_id{
    BRUTE_FORCE,
    FLANN_BASE
};

string type2str(int type);

float computeAndDrawEpiLines(Mat &one, Mat &other, int num_lines, Vec3d &epipole, Mat &fund_mat);

Mat fundamentalMat(Mat &one, Mat &other, vector<Point2d> &good_matches_1, vector<Point2d> &good_matches_2);

pair< vector<Point2f>, vector<Point2f> > match(Mat &one, Mat &other, enum descriptor_id descriptor , enum detector_id detector);

Mat detectFeatures(Mat image, enum detector_id det_id, vector<KeyPoint> &keypoints);

void draw(Mat img, string name);

bool choleskyDecomp(Mat &A, Mat &D);

void obtainAB(const Mat &img, const Mat &mult_mat, Mat &A, Mat &B);

Mat crossProductMatrix(Vec3d elem);

Vec3d maximize(Mat &A, Mat &B);

Vec3d getInitialGuess(Mat &A, Mat &B, Mat &Ap, Mat &Bp);

Mat manualFundMat( vector<Point2d> &good_matches_1,
                    vector<Point2d> &good_matches_2);

double getTranslationTerm(const Mat &img_1, const Mat &img_2, const Mat &H_p,
                          const Mat &Hp_p);

double getMinYCoord(const Mat &img, const Mat &homography);

Mat getS(const Mat &img, const Mat &homography);

void getShearingTransforms(const Mat &img_1, const Mat &img_2,
                          const Mat &H_1, const Mat &H_2,
                          Mat &H_s, Mat &Hp_s);

#endif
