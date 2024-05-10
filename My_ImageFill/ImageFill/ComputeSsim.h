#include"ImageFill.h"
#include <gtest/gtest.h>

double computeSSIM(const cv::Mat& vSourceImage, const cv::Mat& vResultImage);
double getSSIM(const cv::Mat& m_SourceImage, const cv::Mat& m_Result);