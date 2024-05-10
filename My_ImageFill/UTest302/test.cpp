#include "pch.h"
#include"ComputeSsim.h"

class TEST_EdgeFill : public testing::Test
{
protected:
    int HalfPatchWidth = 4;
    int PyramidNumber = 4;
    CImageFill imageFill;
};
static TEST_F(TEST_EdgeFill, NT_EdgeTest)
{
    
    {
        std::string ImageName = "..\\tests\\test1.jpg";
        std::string MaskName = "..\\tests\\test1-mask.jpg";
        double threshold = 0.258473;
        cv::Mat SourceImage = cv::imread(ImageName, cv::IMREAD_COLOR);
        imageFill.initialImage(ImageName, MaskName, HalfPatchWidth, PyramidNumber);
        imageFill.fillCavity();
        double ssim = getSSIM(SourceImage, imageFill.getResult());
        ASSERT_GE(ssim, threshold) << "SSIM value is below threshold.";
    }

    {
        std::string ImageName = "..\\tests\\test2.jpg";
        std::string MaskName = "..\\tests\\test2-mask.jpg";
        double threshold = 0.28556;
        cv::Mat SourceImage = cv::imread(ImageName, cv::IMREAD_COLOR);
        imageFill.initialImage(ImageName, MaskName, HalfPatchWidth, PyramidNumber);
        imageFill.fillCavity();
        double ssim = getSSIM(SourceImage, imageFill.getResult());
        ASSERT_GE(ssim, threshold) << "SSIM value is below threshold.";
    }
}