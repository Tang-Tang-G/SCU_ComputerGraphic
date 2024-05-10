#include "ComputeSsim.h"

double  computeSSIM(cv::Mat& vSourceImage, cv::Mat& vResultImage)
{
    const double C1 = 6.5025, C2 = 58.5225;
    int d = CV_32F;
    cv::Mat I1, I2;
    vSourceImage.convertTo(I1, d);
    vResultImage.convertTo(I2, d);
    cv::Mat I1_2 = I1.mul(I1);
    cv::Mat I2_2 = I2.mul(I2);
    cv::Mat I1_I2 = I1.mul(I2);
    cv::Mat mu1, mu2;
    GaussianBlur(I1, mu1, cv::Size(11, 11), 1.5);
    GaussianBlur(I2, mu2, cv::Size(11, 11), 1.5);
    cv::Mat mu1_2 = mu1.mul(mu1);
    cv::Mat mu2_2 = mu2.mul(mu2);
    cv::Mat mu1_mu2 = mu1.mul(mu2);
    cv::Mat sigma1_2, sigam2_2, sigam12; 
    GaussianBlur(I1_2, sigma1_2, cv::Size(11, 11), 1.5);
    sigma1_2 -= mu1_2;
    GaussianBlur(I2_2, sigam2_2, cv::Size(11, 11), 1.5);
    sigam2_2 -= mu2_2;
    GaussianBlur(I1_I2, sigam12, cv::Size(11, 11), 1.5);
    sigam12 -= mu1_mu2;
    cv::Mat t1, t2, t3;
    t1 = 2 * mu1_mu2 + C1;
    t2 = 2 * sigam12 + C2;
    t3 = t1.mul(t2);
    t1 = mu1_2 + mu2_2 + C1;
    t2 = sigma1_2 + sigam2_2 + C2;
    t1 = t1.mul(t2);
    cv::Mat ssim_map;
    divide(t3, t1, ssim_map); 
    cv::Scalar mssim = mean(ssim_map);
    double ssim = (mssim.val[0] + mssim.val[1] + mssim.val[2]) / 3;
    return ssim;
}

double getSSIM(const cv::Mat& vSourceImage, const cv::Mat& vResult) 
{
    cv::Mat GraySource, GrayResult;
    cv::cvtColor(vSourceImage, GraySource, cv::COLOR_BGR2GRAY);
    cv::cvtColor(vResult, GrayResult, cv::COLOR_BGR2GRAY);
    double ssim = computeSSIM( GraySource,GrayResult);
    return ssim;
}