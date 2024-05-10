#include "pch.h"
#include"ImageFill.h"
class TEST_InitImage :public testing::Test
{
protected:
    CImageFill imageFill;
    int HalfPatchWidth = 4;
    int PyramidNumber = 2;
};

static TEST_F(TEST_InitImage, DE_InvalidInput)
{
    {
        std::string empty_image_name = "";
        std::string empty_mask_name = "..\\tests\\man-mask.png";
        EXPECT_FALSE(imageFill.initialImage(empty_image_name, empty_mask_name, HalfPatchWidth, PyramidNumber));
    }

    {
        std::string empty_mask_name = "..\\tests\man-mask.png";
        std::string non_existing_image_name = "..\\tests\\nonexistent.jpg";
        EXPECT_FALSE(imageFill.initialImage(non_existing_image_name, empty_mask_name, HalfPatchWidth, PyramidNumber));
    }

    {
        std::string existing_image_name = "..\\tests\\man.png";
        std::string non_existing_mask_name = "..\\tests\\nonexistent.jpg";
        EXPECT_FALSE(imageFill.initialImage(existing_image_name, non_existing_mask_name, HalfPatchWidth, PyramidNumber));
    }
}
static TEST_F(TEST_InitImage, NT_ValidInput)
{
    {
        std::string image_name = "..\\tests\\man.png";
        std::string mask_name = "..\\tests\\man-mask.png";
        EXPECT_TRUE(imageFill.initialImage(image_name, mask_name, HalfPatchWidth, PyramidNumber));
        EXPECT_EQ(imageFill.checkValidInputs(), 4);
    }
}
