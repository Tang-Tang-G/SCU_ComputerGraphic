#include "pch.h"
#include"ComputeSsim.h"

class TEST_ImageFill :public testing::Test
{
protected: 
    CImageFill imageFill;
    int HalfPatchWidth = 3;
    int PyramidNumber = 4;
};
static TEST_F(TEST_ImageFill, NT_OneVoid)
{
    {
        std::string ImageName = "..\\tests\\image1.jpg";
        std::string MaskName = "..\\tests\\mask1.jpg";
        double threshold = 0.258473;
        cv::Mat SourceImage = cv::imread(ImageName, cv::IMREAD_COLOR);
        imageFill.initialImage(ImageName, MaskName, HalfPatchWidth, PyramidNumber);
        imageFill.fillCavity();
        double ssim = getSSIM(SourceImage, imageFill.getResult());
        ASSERT_GE(ssim, threshold) << "SSIM value is below threshold.";
    }
}
static TEST_F(TEST_ImageFill, NT_MuchVoids)
{
    {
        std::string ImageName = "..\\tests\\test3.jpg";
        std::string MaskName = "..\\tests\\test3-mask.jpg";
        double threshold = 0.30643;
        imageFill.initialImage(ImageName, MaskName, HalfPatchWidth, PyramidNumber);
        imageFill.fillCavity();
        double ssim = getSSIM(imageFill.getSource(), imageFill.getResult());
        ASSERT_GE(ssim, threshold) << "SSIM value is below threshold.";
    }
    {
        std::string ImageName = "..\\tests\\test4.png";
        std::string MaskName = "..\\tests\\test4-mask.png";
        double threshold = 0.310141;
        imageFill.initialImage(ImageName, MaskName, HalfPatchWidth, PyramidNumber);
        imageFill.fillCavity();
        double ssim = getSSIM(imageFill.getSource(), imageFill.getResult());
        ASSERT_GE(ssim, threshold) << "SSIM value is below threshold.";
    }
}
