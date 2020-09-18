#include <opencv2/opencv.hpp>
#include <iostream>
using namespace std;
using namespace cv;
using namespace cv::dnn;
void main()
{
	cv::dnn::Net unet = cv::dnn::readNetFromONNX("unet.onnx");
	cv::Mat gray;
	Mat img = cv::imread("image.jpg");
	vector<vector<cv::Point>> Contours, EliteContours;
	cv::Mat blob = cv::dnn::blobFromImage(img, 1.0 / 255.f, cv::Size(128, 128), cv::Scalar(), true);
	unet.setInput(blob);
	cv::Mat prob = unet.forward();
	cv::Mat probMat(cv::Size(128, 128), CV_32FC1, prob.ptr());
	probMat.convertTo(probMat, CV_8UC1);
	threshold(probMat, probMat, 1, 255, cv::THRESH_BINARY);

	cv::findContours(probMat, Contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);

	int index;
	int maxArea = 0;
	for (int i = 0; i < Contours.size(); i++)
	{
		if (cv::contourArea(Contours[i]) > 50)
		{
			EliteContours.push_back(Contours[i]);
			index = i;
			maxArea = cv::contourArea(Contours[i]);
		}
	}

	cv::resize(img, img, cv::Size(128, 128));
	cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);
	cv::drawContours(img, Contours, index, cv::Scalar(0, 255, 255), 1);


}