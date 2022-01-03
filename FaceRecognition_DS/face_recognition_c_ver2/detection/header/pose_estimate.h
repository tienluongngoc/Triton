
  
#pragma once
#include <iostream>
#include <stdio.h>
#include <cstdio>
#include "types/facedet.h"
#include "opencv2/objdetect.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/dnn.hpp"
#include "opencv2/dnn/shape_utils.hpp"
#include "opencv2/opencv.hpp"
#include "opencv2/imgproc.hpp"


using namespace std;


// Struct to pass data from DLL
struct TransformData
{
	TransformData(float tx, float ty, float tz, float rfx, float rfy, float rfz, float rux, float ruy, float ruz) :
		tX(tx), tY(ty), tZ(tz), rfX(rfx), rfY(rfy), rfZ(rfz), ruX(rux), ruY(ruy), ruZ(ruz) {}
	float tX, tY, tZ;
	float rfX, rfY, rfZ;
	float ruX, ruY, ruZ;
};



class FaceHeadEstimator
{
public:

	int frame_width;
	int frame_height;
	vector< vector<cv::Point2f> > landmarks;
	std::vector<cv::Point3d> model_points;

public:

	FaceHeadEstimator();
	cv::Vec3f rotationMatrixToEulerAngles(cv::Mat &R);
    void detect(FaceLandmark landmarks,cv::Mat im);
	int init(int& outCameraWidth, int& outCameraHeight);
	void close();
	

};