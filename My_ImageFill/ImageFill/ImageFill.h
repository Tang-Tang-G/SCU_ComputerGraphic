#ifndef IMAGEFILL_H
#define IMAGEFILL_H
#include<opencv2/opencv.hpp>

class CImageFill
{
public:
    bool initialImage(const std::string& vImageName, const std::string& vMaskName, const int vHalfPatchWidth, int PyramidNumber);
    void fillCavity();
    int checkValidInputs();
   const cv::Mat& getSource();
    const cv::Mat& getResult();
    const cv::Mat& getMask();

private:
    cv::Mat m_SourceImage;
    cv::Mat m_Mask;
    cv::Mat m_Result;
    cv::Mat m_WorkImage;
    int m_PyramidNumber;
    int m_HalfPatchWidth;
    const static bool ERROR_INPUT_ImageName_Empty = false;
    const static bool ERROR_INPUT_ImageEmpty = false;
    const static int DEFAULT_HALF_PATCH_WIDTH = 3;
    const static int MODE_ADDITION = 0;
    const static int MODE_MULTIPLICATION = 1;
    const static int ERROR_INPUT_MAT_INVALID_TYPE = 0;
    const static int ERROR_INPUT_MASK_INVALID_TYPE = 1;
    const static int ERROR_MASK_INPUT_SIZE_MISMATCH = 2;
    const static int ERROR_HALF_PATCH_WIDTH_ZERO = 3;
    const static int CHECK_VALID = 4;
};
#endif;

