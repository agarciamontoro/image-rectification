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

enum detector_id{
    ORB,
    BRISK
};

enum descriptor_id{
    BRUTE_FORCE,
    FLANN_BASE
};

float computeAndDrawEpiLines(Mat &one, Mat &other, int num_lines, Vec3d &epipole1, Vec3d &epipole2);

Mat fundamentalMat(Mat &one, Mat &other, vector<Point2d> &good_matches_1, vector<Point2d> &good_matches_2);

pair< vector<Point2f>, vector<Point2f> > match(Mat &one, Mat &other, enum descriptor_id descriptor , enum detector_id detector);

Mat detectFeatures(Mat image, enum detector_id det_id, vector<KeyPoint> &keypoints);

void draw(Mat img, string name);

bool choleskyDecomp(Mat A, Mat &D);

#endif
