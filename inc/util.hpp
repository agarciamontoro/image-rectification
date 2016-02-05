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

#define NEGRO   "\033[1;30m"
#define ROJO    "\033[1;31m"
#define VERDE   "\033[1;32m"
#define AMARILLO "\033[1;33m"
#define AZUL    "\033[1;34m"
#define MAGENTA "\033[1;35m"
#define CYAN    "\033[1;36m"
#define BLANCO  "\033[1;37m"
#define RESET   "\033[0m"


enum detector_id{
    ORB,
    BRISK
};

enum descriptor_id{
    BRUTE_FORCE,
    FLANN_BASE
};

/**
 * Gets a string from Mat type to help debugging http://stackoverflow.com/a/17820615/3370561
 * @param  type Mat.type() integer.
 * @return      String like CV_64F or CV_32UC3.
 */
string type2str(int type);

/**
 * Computes and draw epilines in a stereo pair images.
 * @param  one       First image.
 * @param  other     Second image.
 * @param  num_lines Number of epilines to be drawn.
 * @param  epipole   Output of the epipole.
 * @param  fund_mat  Output of the fundamental mat.
 * @return           Error of the epilines with the correspondences.
 */
double computeAndDrawEpiLines(Mat &one, Mat &other, int num_lines, Vec3d &epipole, Mat &fund_mat);

/**
 * Computes fundamental matrix from 2 images
 * @param  one            First image.
 * @param  other          Second image.
 * @param  good_matches_1 Output of good correspondences from one.
 * @param  good_matches_2 Output of good correspondences from other.
 * @return                Fundamental Matrix.
 */
Mat fundamentalMat(Mat &one, Mat &other, vector<Point2d> &good_matches_1, vector<Point2d> &good_matches_2);

/**
 * Match descriptors from two images.
 * @param  one            First image.
 * @param  other          Second image.
 * @param  descriptor     Matcher to be used.
 * @param  detector       Detector to be used.
 * @return                Pair with correspondences.
 */
pair< vector<Point2d>, vector<Point2d> > match(Mat &one, Mat &other, enum descriptor_id descriptor , enum detector_id detector);

/**
 * Detect features in image
 * @param  image     Input image.
 * @param  det_id    Detector to be used
 * @param  keypoints Output of keypoints.
 * @return           Descriptors from image.
 */
Mat detectFeatures(Mat image, enum detector_id det_id, vector<KeyPoint> &keypoints);

/**
 * Shows an image.
 * @param img  Mat object that will be drawn in a new window.
 * @param name Name of the window that will be created.
 */
void draw(Mat img, string name);


void obtainAB(const Mat &img, const Mat &mult_mat, Mat &A, Mat &B);

/**
 * Returns an antisymmetric matrix representing the cross product with elem.
 * @param  elem Element whose cross product matrix should be computed.
 * @return      A Mat object that represents the antisymmetric matrix associated
 *                with the cross product of the vector elem.
 */
Mat crossProductMatrix(Vec3d elem);

/**
 * Maximize paper equations with A,B input
 * @param  A First Mat.
 * @param  B Second Mat.
 * @return   Z vector with max value.
 */
Vec3d maximize(Mat &A, Mat &B);

/**
 * Get initial guess of z
 * @param  A  A matrix.
 * @param  B  B matrix.
 * @param  Ap A' matrix.
 * @param  Bp B' matrix.
 * @return    Initial guess of z.
 */
Vec3d getInitialGuess(Mat &A, Mat &B, Mat &Ap, Mat &Bp);

/**
 * Manual fundamental Mat for paper example
 * @param  good_matches_1 Matches from first image.
 * @param  good_matches_2 Matches from second image.
 * @return                Fundamental Matrix
 */
Mat manualFundMat( vector<Point2d> &good_matches_1,
                    vector<Point2d> &good_matches_2);

/**
 * Get the v'_c translation for similarity transform.
 * @param  img_1 First image.
 * @param  img_2 Second image.
 * @param  H_p   First proyection.
 * @param  Hp_p  Second proyection.
 * @return       Translation on Y axis.
 */
double getTranslationTerm(const Mat &img_1, const Mat &img_2, const Mat &H_p,
                          const Mat &Hp_p);

/**
 * Get min Y coordinate from homographied image.
 * @param  img        Input image.
 * @param  homography Homography.
 * @return            Min Y coordinate.
 */
double getMinYCoord(const Mat &img, const Mat &homography);

/**
 * Get S from Shearing transform.
 * @param  img        Input image.
 * @param  homography Input homography.
 * @return            S matrix.
 */
Mat getS(const Mat &img, const Mat &homography);

/**
 * Get shearing transform for both images.
 * @param img_1 First image.
 * @param img_2 Second image.
 * @param H_1   First homography.
 * @param H_2   Second homography.
 * @param H_s   H' shearing transform.
 * @param Hp_s  H'_s shearing transform.
 */
void getShearingTransforms(const Mat &img_1, const Mat &img_2,
                          const Mat &H_1, const Mat &H_2,
                          Mat &H_s, Mat &Hp_s);

/**
* Cholesky decomposition.
* @param  A Input matrix.
* @param  L Output upper triangular matrix.
* @return   True if that decomposition exists.
*/
bool choleskyCustomDecomp(const Mat &A, Mat &L);

/**
 * Checks whether an image is inverted after an homography is applied.
 * @param  img        Image to be tested. Mat object.
 * @param  homography Homography to be applied. Mat object.
 * @return            A boolean value showing whether the image would be inverted
 *                      after the homography is applied.
 */
bool isImageInverted(const Mat &img, const Mat &homography);

void optimizeRoot(const Mat &A, const Mat &B,
                  const Mat &Ap, const Mat &Bp,
                  Vec3d &z);

#endif
