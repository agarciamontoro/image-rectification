#include "util.hpp"

string type2str(int type) {
  string r;

  uchar depth = type & CV_MAT_DEPTH_MASK;
  uchar chans = 1 + (type >> CV_CN_SHIFT);

  switch ( depth ) {
    case CV_8U:  r = "8U"; break;
    case CV_8S:  r = "8S"; break;
    case CV_16U: r = "16U"; break;
    case CV_16S: r = "16S"; break;
    case CV_32S: r = "32S"; break;
    case CV_32F: r = "32F"; break;
    case CV_64F: r = "64F"; break;
    default:     r = "User"; break;
  }

  r += "C";
  r += (chans+'0');

  return r;
}


double computeEpiLines(Mat &one, Mat &other, Vec3d &epipole, Mat &fund_mat, vector<Vec3d> &lines_1, vector<Vec3d> &lines_2, vector<Point2d> &good_matches_1,vector<Point2d> &good_matches_2){

    fund_mat = fundamentalMat(one, other, good_matches_1, good_matches_2);
    // fund_mat = manualFundMat(good_matches_1, good_matches_2);

    computeCorrespondEpilines(good_matches_1, 1, fund_mat, lines_2);
    computeCorrespondEpilines(good_matches_2, 2, fund_mat, lines_1);

    // Computes the distances between
    // the lines and their corresponding points.
    double distance_1 = 0.0, distance_2 = 0.0;
    for (size_t i = 0; i < lines_1.size(); i++) {
        Vec2d point_1 = good_matches_1[i];
        Vec2d point_2 = good_matches_2[i];

        Vec3d line_1 = lines_1[i];
        Vec3d line_2 = lines_2[i];

        // Error computation with distance point-to-line
        distance_1 += abs(line_1[0]*point_2[0] +
                          line_1[1]*point_2[1] +
                          line_1[2]) /
                      sqrt(line_1[0]*line_1[0] + line_1[1]*line_1[1]);

        distance_2 += abs(line_2[0]*point_1[0] +
                          line_2[1]*point_1[1] +
                          line_2[2]) /
                      sqrt(line_2[0]*line_2[0] + line_2[1]*line_2[1]);
     }

     // The epipole is the left-null vector of F
     Mat epi_mat;
     SVD::solveZ(fund_mat, epi_mat);

     epipole[0] = epi_mat.at<double>(0,0);
     epipole[1] = epi_mat.at<double>(0,1);
     epipole[2] = epi_mat.at<double>(0,2);

    //  epipole /= epipole[2];

     return (distance_1+distance_2)/(2*lines_1.size());
}


void drawEpilines(Mat &one, Mat &other, vector<Vec3d> &lines_1, vector<Vec3d> &lines_2, vector<Point2d> &good_matches_1, vector<Point2d> &good_matches_2, int num_lines){
  RNG rng;
  theRNG().state = clock();

  for (size_t i = 0; i < lines_1.size(); i++) {
      Vec2d point_1 = good_matches_1[i];
      Vec2d point_2 = good_matches_2[i];

      Vec3d line_1 = lines_1[i];
      Vec3d line_2 = lines_2[i];
    // Draws only num_lines lines
    if(i % (lines_1.size()/num_lines) == 0 ){
        Scalar color(rng.uniform(0, 255),
                     rng.uniform(0, 255),
                     rng.uniform(0, 255));

        line(one,
             Point(0, -line_1[2]/line_1[1]),
             Point(one.cols, -(line_1[2] + line_1[0]*one.cols)/line_1[1]),
             color
             );
        circle(one,
                Point2d(point_1[0], point_1[1]),
                4,
                color,
                CV_FILLED);

        line(other,
             Point(0,
                   -line_2[2]/line_2[1]),
             Point(other.cols,
                   -(line_2[2] + line_2[0]*other.cols)/line_2[1]),
             color
             );
        circle(other,
                Point2d(point_2[0], point_2[1]),
                4,
                color,
                CV_FILLED);

    }
  }
}

Mat fundamentalMat(Mat &one, Mat &other,
                          vector<Point2d> &good_matches_1,
                          vector<Point2d> &good_matches_2){

    pair<vector<Point2d>, vector<Point2d> > matches;
    Mat F;

    matches = match(one, other, descriptor_id::BRUTE_FORCE, detector_id::BRISK);

    vector<unsigned char> mask;
    F = findFundamentalMat(matches.first, matches.second,
                           CV_FM_8POINT | CV_FM_RANSAC,
                           1., 0.99, mask );

    for (size_t i = 0; i < mask.size(); i++) {
        if(mask[i] == 1){
            good_matches_1.push_back(matches.first[i]);
            good_matches_2.push_back(matches.second[i]);
        }
    }

    return F;
}


pair< vector<Point2d>, vector<Point2d> > match(Mat &one, Mat &other, enum descriptor_id descriptor , enum detector_id detector){
    // 1 - Get keypoints and its descriptors in both images
    vector<KeyPoint> keypoints[2];
    Mat descriptors[2];

    descriptors[0] = detectFeatures(one, detector, keypoints[0]);
    descriptors[1] = detectFeatures(other, detector, keypoints[1]);

    // 2 - Match both descriptors using required detector
    // Declare the matcher
    Ptr<DescriptorMatcher> matcher;

    // Define the matcher
    if (descriptor == descriptor_id::BRUTE_FORCE) {
        // For ORB and BRISK descriptors, NORM_HAMMING should be used.
        // See http://sl.ugr.es/norm_ORB_BRISK
        matcher = new BFMatcher(NORM_HAMMING, true);
    }
    else{
        matcher = new FlannBasedMatcher();
        // FlannBased Matcher needs CV_32F descriptors
        // See http://sl.ugr.es/FlannBase_32F
        for (size_t i = 0; i < 2; i++) {
            if (descriptors[i].type() != CV_32F) {
                descriptors[i].convertTo(descriptors[i],CV_32F);
            }
        }
    }

    // Match!
    vector<DMatch> matches;
    matcher->match( descriptors[0], descriptors[1], matches );

    // 3 - Create lists of ordered keypoints following obtained matches
    vector<Point2d> ordered_keypoints[2];

    for( unsigned int i = 0; i < matches.size(); i++ )
    {
      // Get the keypoints from the matches
      ordered_keypoints[0].push_back( keypoints[0][matches[i].queryIdx].pt );
      ordered_keypoints[1].push_back( keypoints[1][matches[i].trainIdx].pt );
    }

    return pair< vector<Point2d>, vector<Point2d> >(ordered_keypoints[0], ordered_keypoints[1]);
}


Mat detectFeatures(Mat image, enum detector_id det_id, vector<KeyPoint> &keypoints){
    // Declare detector
    Ptr<Feature2D> detector;

    // Define detector
    if (det_id == detector_id::ORB) {
        // Declare ORB detector
        detector = ORB::create(
            500,                //nfeatures = 500
            1.2f,               //scaleFactor = 1.2f
            4,                  //nlevels = 8
            21,                 //edgeThreshold = 31
            0,                  //firstLevel = 0
            2,                  //WTA_K = 2
            ORB::HARRIS_SCORE,  //scoreType = ORB::HARRIS_SCORE
            21,                 //patchSize = 31
            20                  //fastThreshold = 20
        );
    }
    else{
        // Declare BRISK and BRISK detectors
        detector = BRISK::create(
            30,   // thresh = 30
    		    3,    // octaves = 3
    		    1.0f  // patternScale = 1.0f
        );
    }

    // Declare array for storing the descriptors
    Mat descriptors;

    // Detect and compute!
    detector->detect(image, keypoints);
    detector->compute(image, keypoints, descriptors);

    return descriptors;
}


void draw(Mat img, string name){
    namedWindow( name, WINDOW_AUTOSIZE );

    // Converts to 8-bits unsigned int to avoid problems
    // in OpenCV implementations in Microsoft Windows.
    Mat image_8U;
    img.convertTo(image_8U, CV_8U);

    imshow( name, image_8U );
}


void obtainAB(const Mat &img, const Mat &mult_mat, Mat &A, Mat &B){
    int width = img.cols;
    int height = img.rows;

    int size = 3;

    Mat PPt = Mat::zeros(size, size, CV_64F);

    PPt.at<double>(0,0) = width*width - 1;
    PPt.at<double>(1,1) = height*height - 1;

    PPt *= (width*height) / 12.0;

    double w_1 = width - 1;
    double h_1 = height - 1;

    double values[3][3] = {
        {w_1*w_1, w_1*h_1, 2*w_1},
        {w_1*h_1, h_1*h_1, 2*h_1},
        {2*w_1, 2*h_1, 4}
    };

    Mat pcpct(size, size, CV_64F, values);

    pcpct /= 4;
    A = mult_mat.t() * PPt * mult_mat;
    B = mult_mat.t() * pcpct * mult_mat;
}


Mat crossProductMatrix(Vec3d elem){
    double values[3][3] = {
        {0, -elem[2], elem[1]},
        {elem[2], 0, -elem[0]},
        {-elem[1], elem[0], 0}
    };

    Mat sol(3, 3, CV_64F, values);

    return sol.clone();
}


Vec3d maximize(Mat &A, Mat &B){
    Mat D; // Output of cholesky decomposition: upper triangular matrix.
    if( choleskyCustomDecomp(A, D) ){

        Mat D_inv = D.inv();

        Mat DBD = D_inv.t() * B * D_inv;

        // Solve the equations system using eigenvalue decomposition

        Mat eigenvalues, eigenvectors;
        eigen(DBD, eigenvalues, eigenvectors);

        // Take largest eigenvector
        Mat y = eigenvectors.row(0);

        Mat sol = D_inv*y.t();

        Vec3d res(sol.at<double>(0,0), sol.at<double>(1,0), sol.at<double>(2,0));

        return res;
    }

    // At this point, there is an error!
    Mat eigenvalues;
    eigen(A, eigenvalues);

    cout << "\n\n\n----------------------------- WARNING -----------------------" << endl;
    cout << "A = " << A << endl;
    cout << "A eigenvalues: " << eigenvalues << endl << endl;

    return Vec3d(0, 0, 0);
}


Vec3d getInitialGuess(Mat &A, Mat &B, Mat &Ap, Mat &Bp){

    Vec3d z_1 = maximize(A, B);
    Vec3d z_2 = maximize(Ap, Bp);

    cout << "Z_1: " << z_1 << endl <<"Normalizado: " << normalize(z_1) << endl;
    cout << "Z_2: " << z_2 << endl <<"Normalizado: " << normalize(z_2) << endl;

    return (normalize(z_1) + normalize(z_2))/2;
}


Mat manualFundMat( vector<Point2d> &good_matches_1,
                    vector<Point2d> &good_matches_2){
    // Taking points by hand
    vector<Point> origin, destination;

    origin.push_back(Point(63, 31));
    origin.push_back(Point(69, 39));
    origin.push_back(Point(220, 13));
    origin.push_back(Point(444, 23));
    origin.push_back(Point(355, 45));
    origin.push_back(Point(347, 55));
    origin.push_back(Point(80, 319));
    origin.push_back(Point(85, 313));
    origin.push_back(Point(334, 371));
    origin.push_back(Point(342, 381));

    origin.push_back(Point(213, 126));
    origin.push_back(Point(298, 158));
    origin.push_back(Point(219, 266));

    destination.push_back(Point(159, 51));
    destination.push_back(Point(167, 59));
    destination.push_back(Point(81, 28));
    destination.push_back(Point(293, 20));
    destination.push_back(Point(440, 38));
    destination.push_back(Point(435, 45));
    destination.push_back(Point(171, 372));
    destination.push_back(Point(178, 363));
    destination.push_back(Point(420, 305));
    destination.push_back(Point(424, 311));

    destination.push_back(Point(188, 140));
    destination.push_back(Point(235, 156));
    destination.push_back(Point(202, 278));

    vector<unsigned char> mask;
    Mat fund_mat = findFundamentalMat(origin, destination,
                           CV_FM_8POINT | CV_FM_RANSAC,
                           20, 0.99, mask );

   for (size_t i = 0; i < mask.size(); i++) {
       if(/*mask[i] == 1*/true){
           good_matches_1.push_back(origin[i]);
           good_matches_2.push_back(destination[i]);
       }
   }

    return fund_mat;
}


double getTranslationTerm(const Mat &img_1, const Mat &img_2, const Mat &H_p, const Mat &Hp_p){
    double min_1 = getMinYCoord(img_1, H_p);
    double min_2 = getMinYCoord(img_2, Hp_p);

    double offset = min_1 < min_2 ? min_1 : min_2;

    return -offset;
}


double getMinYCoord(const Mat &img, const Mat &homography){
    vector<Point2d> corners(4), corners_trans(4);

    corners[0] = Point2d(0,0);
    corners[1] = Point2d(img.cols,0);
    corners[2] = Point2d(img.cols,img.rows);
    corners[3] = Point2d(0,img.rows);

    perspectiveTransform(corners, corners_trans, homography);

    double min_y;
    min_y = +INF;

    for (int j = 0; j < 4; j++) {
        min_y = min(corners_trans[j].y, min_y);
    }

    return min_y;
}


Mat getS(const Mat &img, const Mat &homography){
    int w = img.cols;
    int h = img.rows;

    Point2d a((w-1)/2, 0);
    Point2d b(w-1, (h-1)/2);
    Point2d c((w-1)/2, h-1);
    Point2d d(0, (h-1)/2);

    vector<Point2d> midpoints, midpoints_hat;
    midpoints.push_back(a);
    midpoints.push_back(b);
    midpoints.push_back(c);
    midpoints.push_back(d);

    perspectiveTransform(midpoints, midpoints_hat, homography);

    Point2d x = midpoints_hat[1] - midpoints_hat[3];
    Point2d y = midpoints_hat[2] - midpoints_hat[0];

    double coeff_a = (h*h*x.y*x.y + w*w*y.y*y.y) / (h*w * (x.y*y.x - x.x*y.y));
    double coeff_b = (h*h*x.x*x.y + w*w*y.x*y.y) / (h*w * (x.x*y.y - x.y*y.x));

    Mat S = Mat::eye(3, 3, CV_64F);
    S.at<double>(0,0) = coeff_a;
    S.at<double>(0,1) = coeff_b;

    Vec3d x_hom(x.x, x.y, 0.0);
    Vec3d y_hom(y.x, y.y, 0.0);

    if( coeff_a < 0 ){
        coeff_a *= -1;
        coeff_b *= -1;

        S.at<double>(0,0) = coeff_a;
        S.at<double>(0,1) = coeff_b;
    }


    Mat EQ18 = (S * Mat(x_hom)).t() * (S * Mat(y_hom));
    cout << ROJO << "EQ18 " << EQ18 << RESET << endl;

    Mat EQ19 = ((S * Mat(x_hom)).t() * (S * Mat(x_hom))) / ((S * Mat(y_hom)).t() * (S * Mat(y_hom))) - (1.*w*w)/(1.*h*h);
    cout << ROJO << "EQ19 " << EQ19 << RESET << endl;

    cout << "S = " << S << endl;

    return S;
}


void getShearingTransforms(const Mat &img_1, const Mat &img_2,
                           const Mat &H_1, const Mat &H_2,
                           Mat &H_s, Mat &Hp_s){

    Mat S = getS(img_1, H_1);
    Mat Sp = getS(img_2, H_2);

    double A = img_1.cols*img_1.rows + img_2.cols*img_2.rows;
    double Ap = 0;

    vector<Point2f> corners(4), corners_trans(4);

    corners[0] = Point2f(0,0);
    corners[1] = Point2f(img_1.cols,0);
    corners[2] = Point2f(img_1.cols,img_1.rows);
    corners[3] = Point2f(0,img_1.rows);

    perspectiveTransform(corners, corners_trans, S*H_1);
    Ap += contourArea(corners_trans);

    float min_x_1, min_y_1;
    min_x_1 = min_y_1 = +INF;
    for (int j = 0; j < 4; j++) {
        min_x_1 = min(corners_trans[j].x, min_x_1);
        min_y_1 = min(corners_trans[j].y, min_y_1);
    }

    corners[0] = Point2f(0,0);
    corners[1] = Point2f(img_2.cols,0);
    corners[2] = Point2f(img_2.cols,img_2.rows);
    corners[3] = Point2f(0,img_2.rows);

    perspectiveTransform(corners, corners_trans, Sp*H_2);
    Ap += contourArea(corners_trans);

    float min_x_2, min_y_2;
    min_x_2 = min_y_2 = +INF;
    for (int j = 0; j < 4; j++) {
        min_x_2 = min(corners_trans[j].x, min_x_2);
        min_y_2 = min(corners_trans[j].y, min_y_2);
    }

    double scale = sqrt(A/Ap);
    cout << "A = " << A << " \n/ Ap = " << Ap << endl;

    double min_y = min_y_1 < min_y_2 ? min_y_1 : min_y_2;

    // We define W2 as the scale transformation and W1 as the translation
    // transformation. Then, W = W1*W2.

    Mat W;
    Mat Wp;

    Mat W_1 = Mat::eye(3, 3, CV_64F);
    Mat Wp_1 = Mat::eye(3, 3, CV_64F);

    Mat W_2 = Mat::eye(3, 3, CV_64F);
    Mat Wp_2 = Mat::eye(3, 3, CV_64F);

    W_2.at<double>(0,0) = W_2.at<double>(1,1) = scale;
    Wp_2.at<double>(0,0) = Wp_2.at<double>(1,1) = scale;

    if(isImageInverted(img_1, W_2*H_1)){
      W_2.at<double>(0,0) = W_2.at<double>(1,1) = -scale;
      Wp_2.at<double>(0,0) = Wp_2.at<double>(1,1) = -scale;
    }

            corners[0] = Point2d(0,0);
            corners[1] = Point2d(img_1.cols,0);
            corners[2] = Point2d(img_1.cols,img_1.rows);
            corners[3] = Point2d(0,img_1.rows);

            perspectiveTransform(corners, corners_trans, W_2*S*H_1);

            min_x_1 = min_y_1 = +INF;
            for (int j = 0; j < 4; j++) {
                min_x_1 = min(corners_trans[j].x, min_x_1);
                min_y_1 = min(corners_trans[j].y, min_y_1);
            }

            corners[0] = Point2d(0,0);
            corners[1] = Point2d(img_2.cols,0);
            corners[2] = Point2d(img_2.cols,img_2.rows);
            corners[3] = Point2d(0,img_2.rows);

            perspectiveTransform(corners, corners_trans, Wp_2*Sp*H_2);

            min_x_2 = min_y_2 = +INF;
            for (int j = 0; j < 4; j++) {
                min_x_2 = min(corners_trans[j].x, min_x_2);
                min_y_2 = min(corners_trans[j].y, min_y_2);
            }

            min_y = min_y_1 < min_y_2 ? min_y_1 : min_y_2;

    W_1.at<double>(0,2) = -min_x_1;
    Wp_1.at<double>(0,2) = -min_x_2;

    W_1.at<double>(1,2) = Wp_1.at<double>(1,2) = -min_y;

    W = W_1*W_2;
    Wp = Wp_1*Wp_2;

    H_s = W*S;
    Hp_s = Wp*Sp;

    cout << "H_s = " << H_s << "\nHp_s = " << Hp_s << endl;
}


bool choleskyCustomDecomp(const Mat &A, Mat &L){

  L = Mat::zeros(3,3,CV_64F);

  for (int i = 0; i < 3; i++){
    for (int j = 0; j <= i; j++){
      double sum = 0;
      for (int k = 0; k < j; k++){
        sum += L.at<double>(i,k) * L.at<double>(j,k);
      }

      L.at<double>(i,j) = A.at<double>(i,j) - sum;
      if (i == j){
        if (L.at<double>(i,j) < 0.0){
          if (L.at<double>(i,j) > -1e-5){
            L.at<double>(i,j) *= -1;
          }
          else{
            cout << "ERROR HERE HERE HERE: " << L.at<double>(i,j) << endl;
            return false;
          }
        }
        L.at<double>(i,j) = sqrt(L.at<double>(i,j));
      }
      else{
        L.at<double>(i,j) /= L.at<double>(j,j);
      }
    }
  }

  L = L.t();

  return true;
}



bool isImageInverted(const Mat &img, const Mat &homography){
  vector<Point2d> corners(2), corners_trans(2);

  corners[0] = Point2d(0,0);
  corners[1] = Point2d(0,img.rows);

  perspectiveTransform(corners, corners_trans, homography);

  return corners_trans[1].y - corners_trans[0].y < 0.0;
}

vector< vector<double> > MatToVector(const Mat &mat){
    vector< vector<double> > array;

    for (size_t i = 0; i < mat.rows; i++) {
        vector<double> row;

        for (size_t j = 0; j < mat.cols; j++) {
            row.push_back(mat.at<double>(i,j));
        }

        array.push_back(row);
    }

    return array;
}


double function(const Mat &A, const Mat &B,
                const Mat &Ap, const Mat &Bp,
                double x){
    vector< vector<double> > a = MatToVector(A);
    vector< vector<double> > b = MatToVector(B);
    vector< vector<double> > ap = MatToVector(Ap);
    vector< vector<double> > bp = MatToVector(Bp);

    double summ_1 =
    (2*ap[0][0]*x+ap[1][0]+ap[0][1])/(x*(bp[0][0]*x+bp[0][1])+bp[1][0]*x+bp[1][1]);

    double den_summ_2 = x*(bp[0][0]*x+bp[0][1])+bp[1][0]*x+bp[1][1];
    den_summ_2 = den_summ_2*den_summ_2;

    double summ_2 =
    ((2*bp[0][0]*x+bp[1][0]+bp[0][1])*(x*(ap[0][0]*x+ap[0][1])+ap[1][0]*x+ap[1][1]))/den_summ_2;

    double summ_3 =
    (2*a[0][0]*x+a[1][0]+a[0][1])/(x*(b[0][0]*x+b[0][1])+b[1][0]*x+b[1][1]);

    double den_summ_4 = x*(b[0][0]*x+b[0][1])+b[1][0]*x+b[1][1];
    den_summ_4 = den_summ_4 * den_summ_4;

    double summ_4 =
    ((2*b[0][0]*x+b[1][0]+b[0][1])*(x*(a[0][0]*x+a[0][1])+a[1][0]*x+a[1][1]))/den_summ_4;

    return summ_1 - summ_2 + summ_3 - summ_4;
}


double derivative(const Mat &A, const Mat &B,
                  const Mat &Ap, const Mat &Bp,
                  double x){
    vector< vector<double> > a = MatToVector(A);
    vector< vector<double> > b = MatToVector(B);
    vector< vector<double> > ap = MatToVector(Ap);
    vector< vector<double> > bp = MatToVector(Bp);


    double summ_1 = (2*ap[0][0])/(x*(bp[0][0]*x+bp[0][1])+bp[1][0]*x+bp[1][1]);

    double den_summ_2 = (x*(bp[0][0]*x+bp[0][1])+bp[1][0]*x+bp[1][1]);
    den_summ_2 = den_summ_2 * den_summ_2;

    double summ_2 =
    (2*bp[0][0]*(x*(ap[0][0]*x+ap[0][1])+ap[1][0]*x+ap[1][1]))/den_summ_2;

    double den_summ_3 = (x*(bp[0][0]*x+bp[0][1])+bp[1][0]*x+bp[1][1]);
    den_summ_3 = den_summ_3 * den_summ_3;

    double summ_3 =
    (2*(2*ap[0][0]*x+ap[1][0]+ap[0][1])*(2*bp[0][0]*x+bp[1][0]+bp[0][1]))/den_summ_3;

    double den_summ_4 = (x*(bp[0][0]*x+bp[0][1])+bp[1][0]*x+bp[1][1]);
    den_summ_4 = den_summ_4 * den_summ_4 * den_summ_4;

    double aux_num_summ_4 = (2*bp[0][0]*x+bp[1][0]+bp[0][1]);
    aux_num_summ_4 = aux_num_summ_4 * aux_num_summ_4;

    double summ_4 =
    (2*aux_num_summ_4*(x*(ap[0][0]*x+ap[0][1])+ap[1][0]*x+ap[1][1]))/den_summ_4;

    double summ_5 =  (2*a[0][0])/(x*(b[0][0]*x+b[0][1])+b[1][0]*x+b[1][1]);


    double den_summ_6 = (x*(b[0][0]*x+b[0][1])+b[1][0]*x+b[1][1]);
    den_summ_6 = den_summ_6 * den_summ_6;

    double summ_6 =
    (2*b[0][0]*(x*(a[0][0]*x+a[0][1])+a[1][0]*x+a[1][1]))/den_summ_6;

    double den_summ_7 = (x*(b[0][0]*x+b[0][1])+b[1][0]*x+b[1][1]);
    den_summ_7 = den_summ_7 * den_summ_7;

    double summ_7 =
    (2*(2*a[0][0]*x+a[1][0]+a[0][1])*(2*b[0][0]*x+b[1][0]+b[0][1]))/den_summ_7;

    double den_summ_8 = (x*(b[0][0]*x+b[0][1])+b[1][0]*x+b[1][1]);
    den_summ_8 = den_summ_8 * den_summ_8 * den_summ_8;

    double aux_num_summ_8 = (2*b[0][0]*x+b[1][0]+b[0][1]);
    aux_num_summ_8 = aux_num_summ_8 * aux_num_summ_8;

    double summ_8 =
    (2*aux_num_summ_8*(x*(a[0][0]*x+a[0][1])+a[1][0]*x+a[1][1]))/den_summ_8;

    return summ_1 - summ_2 - summ_3 + summ_4 + summ_5 - summ_6 - summ_7 + summ_8;
}


double NewtonRaphson(const Mat &A, const Mat &B,
                     const Mat &Ap, const Mat &Bp,
                     double init_guess){
    double current = init_guess;
    double previous;

    double fx = function(A,B,Ap,Bp, current);
    double dfx = derivative(A,B,Ap,Bp, current);

    do {
        previous = current;
        current = current - fx / dfx;

        fx = function(A,B,Ap,Bp, current);
        dfx = derivative(A,B,Ap,Bp, current);

        cout << ROJO << fx << RESET << endl;
    } while (abs(fx) > 1e-15);
    // Double-precision values have 15 stable decimal positions

    return current;
}


void optimizeRoot(const Mat &A, const Mat &B,
                   const Mat &Ap, const Mat &Bp,
                   Vec3d &z){

    double lambda = z[0];

    z[0] = NewtonRaphson(A,B,Ap,Bp, lambda);
    z[1] = 1.0;
    z[2] = 0.0;
}
