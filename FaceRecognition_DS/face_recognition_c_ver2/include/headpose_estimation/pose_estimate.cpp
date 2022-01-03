#pragma once
#include "pose_estimate.h"

using namespace std;

Estimator::Estimator()
{

    // Average human face 3d points (Nose IS longer than average)
    model_points.push_back(cv::Point3d(0.0f, 0.0f, 60.f));          // Nose tip
    model_points.push_back(cv::Point3d(0.0f, -330.0f, -65.0f));     // Chin
    model_points.push_back(cv::Point3d(-225.0f, 170.0f, -135.0f));  // Left eye left corner
    model_points.push_back(cv::Point3d(225.0f, 170.0f, -135.0f));   // Right eye right corner
    model_points.push_back(cv::Point3d(-150.0f, -150.0f, -125.0f)); // Left Mouth corner
    model_points.push_back(cv::Point3d(150.0f, -150.0f, -125.0f));  // Right mouth corner
    // Set resolution

}

int Estimator::init(int &outCameraWidth, int &outCameraHeight)
{
    frame_width = outCameraWidth;
    frame_height = outCameraHeight;
}

void Estimator::close()
{
}

void Estimator::detect(face_landmark landmarks)
{

    // Prepair face points for perspective solve
    vector<cv::Point2d> image_points;

    float v_cx = landmarks.x[2] - (landmarks.x[0] + landmarks.x[1]) / 2;
    float v_cy = landmarks.y[2] - (landmarks.y[0] + landmarks.y[1]) / 2;
    image_points.push_back(cv::Point2d(landmarks.x[2], landmarks.y[2]));                           // Nose tip
    image_points.push_back(cv::Point2d(landmarks.x[2] + 1.2 * v_cx, landmarks.y[2] + 1.2 * v_cy)); // Chin
    image_points.push_back(cv::Point2d(landmarks.x[0], landmarks.y[0]));                           // Left eye left corner
    image_points.push_back(cv::Point2d(landmarks.x[1], landmarks.y[1]));                           // Right eye right corner
    image_points.push_back(cv::Point2d(landmarks.x[3], landmarks.y[3]));                           // Left Mouth corner
    image_points.push_back(cv::Point2d(landmarks.x[4], landmarks.y[4]));                           // Right mouth corner

    // Generate fake camera Matrix
    double focal_length = frame_width;
    cv::Point2d center = cv::Point2d(frame_width / 2, frame_height / 2);
    cv::Mat camera_matrix = (cv::Mat_<double>(3, 3) << focal_length, 0, center.x, 0, focal_length, center.y, 0, 0, 1);
    cv::Mat dist_coeffs = cv::Mat::zeros(4, 1, cv::DataType<double>::type);

    // Output rotation and translation
    cv::Mat rotation_vector;
    cv::Mat translation_vector;
    cv::Mat rot_mat;

    // Solve for pose
    cv::solvePnP(model_points, image_points, camera_matrix, dist_coeffs, rotation_vector, translation_vector);

    // Convert rotation to Matrix
    cv::Rodrigues(rotation_vector, rot_mat);

    // Export transform
    TransformData(translation_vector.at<double>(0), translation_vector.at<double>(1), translation_vector.at<double>(2),
                             rot_mat.at<double>(2, 0), rot_mat.at<double>(2, 1), rot_mat.at<double>(2, 2),
                             rot_mat.at<double>(1, 0), rot_mat.at<double>(1, 1), rot_mat.at<double>(1, 2));
}
