#include "ImageFill.h"
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/types_c.h>
#include <vector>
using namespace cv;

void patchMatch(const Mat& vSourceImage, const Mat& vTargetImage, const Mat& vMask, int vPatchSize, Mat& vNearestNeighbor);
Vec3f meanShift(std::vector<Vec3b> &vVecVoteColor, std::vector<float> &vVecVoteWeight, int vSigma);

const cv::Mat& CImageFill::getSource()
{
	return m_SourceImage;
}

const cv::Mat& CImageFill::getResult()
{
	return m_Result;
}

const cv::Mat& CImageFill::getMask()
{
	return m_Mask;
}

bool CImageFill::initialImage(const std::string& vImageName, const std::string& vMaskName, int vHalfPatchWidth, int vPyramidNumber)
{
	if (vImageName.empty() || vMaskName.empty())
	{
		return ERROR_INPUT_ImageName_Empty;
	}
    m_SourceImage = cv::imread(vImageName, cv::IMREAD_COLOR);
	m_Mask = cv::imread(vMaskName, 0);
	if (m_SourceImage.empty() || m_Mask.empty())
	{
		return ERROR_INPUT_ImageEmpty;
	}
	m_HalfPatchWidth = vHalfPatchWidth;
	m_WorkImage = m_SourceImage.clone();
	m_Result.create(m_SourceImage.size(), m_SourceImage.type());
	m_PyramidNumber = vPyramidNumber;
	return true;
}

int CImageFill::checkValidInputs()
{
	if (m_SourceImage.type()!=CV_8UC3)
		return ERROR_INPUT_MAT_INVALID_TYPE;
	else if (m_Mask.type() != CV_8UC1)
		return ERROR_INPUT_MASK_INVALID_TYPE;
	else if (!CV_ARE_SIZES_EQ(&m_Mask, &m_SourceImage))
		return ERROR_MASK_INPUT_SIZE_MISMATCH;
	else if (m_HalfPatchWidth == 0)
		return ERROR_HALF_PATCH_WIDTH_ZERO;
	 return CHECK_VALID;
}

void CImageFill::fillCavity()
{
	if (checkValidInputs() == CHECK_VALID)
	{
		Mat Weights = Mat(m_WorkImage.size(), CV_32F);
		distanceTransform(m_Mask, Weights, DIST_L2, 3);
		for (int i = 0; i < m_WorkImage.rows; i++)
		{
			for (int j = 0; j < m_WorkImage.cols; j++)
			{
				float Distance = Weights.at<float>(i, j);
				Weights.at<float>(i, j) = (float)pow(1.3, -Distance);
				if (m_Mask.at<uchar>(i, j))
				{
					m_SourceImage.at<Vec3b>(i, j)[0] = rand()%255;
					m_SourceImage.at<Vec3b>(i, j)[1] = rand()%255;
					m_SourceImage.at<Vec3b>(i, j)[2] = rand()%255;
				}
			}
		}
		Mat CurImage = Mat();
		Mat CurMask;
		Mat CurWeight;
		int PatchSize = 2 * m_HalfPatchWidth + 1;
		float Convergence = 10.0f;

		while (m_PyramidNumber >=0)
		{
			float Scale = 1.0f / (1 << m_PyramidNumber);
			if (m_PyramidNumber <=0)
			{
				Scale = 1.0f;
				m_HalfPatchWidth /= 2;
				PatchSize = 2 * m_HalfPatchWidth + 1;
			}
			resize(m_SourceImage, m_WorkImage, Size(m_SourceImage.cols * Scale, Scale * m_SourceImage.rows));
			resize(m_Mask, CurMask, Size(m_Mask.cols * Scale, Scale * m_Mask.rows));
			resize(Weights, CurWeight, Size(Weights.cols * Scale, Scale * Weights.rows));

			if (CurImage.rows * CurImage.cols > 0)
			{
				resize(CurImage, CurImage, Size(m_SourceImage.cols * Scale, Scale * m_SourceImage.rows));
				for (int i = 0; i < CurImage.rows; i++)
					for (int j = 0; j < CurImage.cols; j++)
						if (CurMask.at<uchar>(i, j)==0)
							CurImage.at<Vec3b>(i, j) = m_WorkImage.at<Vec3b>(i, j);
			}
			else
				CurImage = m_WorkImage.clone();

			int IterMaxNumber = 20;
			if (m_PyramidNumber <=1)
				IterMaxNumber = 10;
			int MinLength = min(CurImage.rows, CurImage.cols);
			if (MinLength < 2 * PatchSize)
			{
				m_PyramidNumber--;
				continue;
			}
			while (IterMaxNumber--)
			{
				Mat LastImage = CurImage.clone();
				Mat NNF;
				patchMatch(CurImage, CurImage, CurMask, PatchSize, NNF);

				for (int i = 0; i < CurImage.rows; i++)
				{
					for (int k = 0; k < CurImage.cols; k++)
					{
						if (CurMask.at<uchar>(i, k))
						{
							std::vector<Vec3b> VoteColor;
							std::vector<float> VoteWeight;
							std::vector<float> Dist;

							for (int OffsetX = -PatchSize + 1; OffsetX <= 0; OffsetX++)
							{
								for (int OffsetY = -PatchSize + 1; OffsetY <= 0; OffsetY++)
								{
									int StartX = i + OffsetX;
									int StartY = k + OffsetY;
									if (StartX < 0 || StartY < 0 || StartY + PatchSize >= CurImage.cols - 1 || StartX + PatchSize >= CurImage.rows - 1) continue;
									Rect CurPatchRect(StartY, StartX, PatchSize, PatchSize);  // 当前块的
									Mat CurPatch = CurImage(CurPatchRect);
									int NNF_X = NNF.at<Vec3i>(StartX, StartY)[0];
									int NNF_Y = NNF.at<Vec3i>(StartX, StartY)[1];
									Rect NearPatchRect(NNF_X, NNF_Y, PatchSize, PatchSize);  // 当前块的
									Mat NearestPatch = CurImage(NearPatchRect);
									float Distance = norm(CurPatch, NearestPatch);
									Vec3b Color = NearestPatch.at<Vec3b>(-OffsetX, -OffsetY);
									float Weight = Weights.at<float>(StartX + PatchSize / 2, StartY + PatchSize / 2);
									Dist.push_back(Distance * Distance);
									VoteColor.push_back(Color);
									VoteWeight.push_back(Weight);
								}
							}
							if (VoteWeight.size() < 3)
							{
								continue;
							}
							std::vector<float> DistCopy;
							DistCopy.assign(Dist.begin(), Dist.end());
							sort(DistCopy.begin(), DistCopy.end());

							float Sigma = DistCopy[DistCopy.size() * 3 / 4];
							for (int i = 0; i < VoteWeight.size(); i++)
							{
								if (Sigma != 0)
									VoteWeight[i] = VoteWeight[i] * exp(-(Dist[i]) / (2 * Sigma));
							}

							float MaxWeight = 0;
							for (int i = 0; i < VoteWeight.size(); i++)
							{
								if (MaxWeight < VoteWeight[i])
								{
									MaxWeight = VoteWeight[i];
								}
							}
							for (int i = 0; i < VoteWeight.size(); i++)
							{
								VoteWeight[i] = VoteWeight[i] / MaxWeight;
							}

							CurImage.at<Vec3b>(i, k) = meanShift(VoteColor, VoteWeight, 50);
						}
					}
				}
				float Diff = 0;
				int Num = 0;
				for (int i = 0; i < LastImage.rows; i++)
				{
					for (int j = 0; j < LastImage.cols; j++)
					{
						if (CurMask.at<uchar>(i, j))
						{
							Vec3f a1 = LastImage.at<Vec3b>(i, j);
							Vec3f a2 = CurImage.at<Vec3b>(i, j);
							Vec3f a3 = a1 - a2;
							Diff += a3[0] * a3[0] + a3[1] * a3[1] + a3[2] * a3[2];
							Num++;
						}
					}
				}
				Diff = Diff / Num;
					imshow("CurImage", CurImage);
					waitKey(100);
					printf("Pyramid: %d, scale: %f, Diff: %f,IterMaxNumber:%d \n", m_PyramidNumber, Scale, Diff,IterMaxNumber);
				if (Diff < Convergence)	break;
			}
			m_PyramidNumber--;
		}
		m_Result = CurImage;
	}
}

