#pragma once
#include "header/pose_estimate.h"

using namespace std;

FaceHeadEstimator::FaceHeadEstimator() {

  // Average human face 3d points (Nose IS longer than average)
  model_points.push_back(cv::Point3d(0.0f, 0.0f, 60.f));      // Nose tip
  model_points.push_back(cv::Point3d(0.0f, -330.0f, -65.0f)); // Chin
  model_points.push_back(
      cv::Point3d(-225.0f, 170.0f, -135.0f)); // Left eye left corner
  model_points.push_back(
      cv::Point3d(225.0f, 170.0f, -135.0f)); // Right eye right corner
  model_points.push_back(
      cv::Point3d(-150.0f, -150.0f, -125.0f)); // Left Mouth corner
  model_points.push_back(
      cv::Point3d(150.0f, -150.0f, -125.0f)); // Right mouth corner
                                              // Set resolution
}

int FaceHeadEstimator::init(int &outCameraWidth, int &outCameraHeight) {
  frame_width = outCameraWidth;
  frame_height = outCameraHeight;
}

void FaceHeadEstimator::close() {}

cv::Vec3f FaceHeadEstimator::rotationMatrixToEulerAngles(cv::Mat &R) {

  float sy = sqrt(R.at<double>(0, 0) * R.at<double>(0, 0) +
                  R.at<double>(1, 0) * R.at<double>(1, 0));

  bool singular = sy < 1e-6; // If

  float x, y, z;
  if (!singular) {
    x = atan2(R.at<double>(2, 1), R.at<double>(2, 2));
    y = atan2(-R.at<double>(2, 0), sy);
    z = atan2(R.at<double>(1, 0), R.at<double>(0, 0));
  } else {
    x = atan2(-R.at<double>(1, 2), R.at<double>(1, 1));
    y = atan2(-R.at<double>(2, 0), sy);
    z = 0;
  }
  return cv::Vec3f(x, y, z);
}
float clip(float a,float mina,float minb){
  if(a<mina) a=mina;
  if(a>minb) a=minb;
  return a;
}

void FaceHeadEstimator::detect(FaceLandmark landmarks, cv::Mat im) {

  // Prepair face points for perspective solve
  vector<cv::Point2d> image_points;

  float v_cx = landmarks.x[2] - (landmarks.x[0] + landmarks.x[1]) / 2;
  float v_cy = landmarks.y[2] - (landmarks.y[0] + landmarks.y[1]) / 2;
  image_points.push_back(
      cv::Point2d(landmarks.x[2], landmarks.y[2])); // Nose tip
  image_points.push_back(cv::Point2d(landmarks.x[2] + 1.2 * v_cx,
                                     landmarks.y[2] + 1.2 * v_cy)); // Chin
  image_points.push_back(
      cv::Point2d(landmarks.x[0], landmarks.y[0])); // Left eye left corner
  image_points.push_back(
      cv::Point2d(landmarks.x[1], landmarks.y[1])); // Right eye right corner
  image_points.push_back(
      cv::Point2d(landmarks.x[3], landmarks.y[3])); // Left Mouth corner
  image_points.push_back(
      cv::Point2d(landmarks.x[4], landmarks.y[4])); // Right mouth corner

  // Generate fake camera Matrix
  double focal_length = frame_width;
  cv::Point2d center = cv::Point2d(frame_width / 2, frame_height / 2);
  cv::Mat camera_matrix = (cv::Mat_<double>(3, 3) << focal_length, 0, center.x,
                           0, focal_length, center.y, 0, 0, 1);

  cv::Mat dist_coeffs = cv::Mat::zeros(4, 1, cv::DataType<double>::type);

  // Output rotation and translation
  cv::Mat rotation_vector;
  cv::Mat translation_vector;
  cv::Mat rot_mat;

  // Solve for pose
  cv::solvePnP(model_points, image_points, camera_matrix, dist_coeffs,
               rotation_vector, translation_vector);
  vector<cv::Point3d> nose_end_point3D;
  vector<cv::Point2d> nose_end_point2D;
  nose_end_point3D.push_back(cv::Point3d(0, 0, 1000.0));

  cv::projectPoints(nose_end_point3D, rotation_vector, translation_vector,
                    camera_matrix, dist_coeffs, nose_end_point2D);
  //   cout << "Rotation Vector "
  //        << " : " << rotation_vector << endl;
  cv::Vec3f res = rotationMatrixToEulerAngles(rotation_vector);
  res[0]=clip(res[0]*1.25,-1.0,1.0);
  res[1]=clip(res[1]*1.25,-1.0,1.0);
  res[2]=clip(res[2]*1.25,-1.0,1.0);


  res[0] *= (180.0 / 3.141592653589793238463);
  res[1] *= (180.0 / 3.141592653589793238463);
  res[2] *= (180.0 / 3.141592653589793238463);
  std::cout << "angle " << res[0] << " " << res[1] << " " << res[2] << "\n";
  //   cout << "Translation Vector"
  //        << " : " << translation_vector << endl;
  //   cout << nose_end_point2D << endl;

  for (int i = 0; i < image_points.size(); i++) {
    circle(im, image_points[i], 3, cv::Scalar(0, 0, 255), -1);
  }
  cv::line(im, image_points[0], nose_end_point2D[0], cv::Scalar(255, 0, 0), 2);

  // Convert rotation to Matrix
  cv::Rodrigues(rotation_vector, rot_mat);

  // Export transform
  // std::cout << "angle 2 : " << translation_vector.at<double>(0) << " "
  //           << translation_vector.at<double>(1) << " "
  //           << translation_vector.at<double>(2) <<"\n";

  TransformData(translation_vector.at<double>(0),
                translation_vector.at<double>(1),
                translation_vector.at<double>(2), rot_mat.at<double>(2, 0),
                rot_mat.at<double>(2, 1), rot_mat.at<double>(2, 2),
                rot_mat.at<double>(1, 0), rot_mat.at<double>(1, 1),
                rot_mat.at<double>(1, 2));
}
