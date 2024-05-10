#include "ImageFill.h"

using namespace cv;

cv::Vec3f meanShift(std::vector<cv::Vec3b> &vColors, std::vector<float> &vWeights, int vSigma)
{
	Vec3f Mean3f = Vec3f{ 0,0,0 };
	float TotalWeight = 0.0f;
	for (int i = 0; i < vColors.size(); i++)
	{
		Mean3f += (Vec3f)(vColors[i] * vWeights[i]);
		TotalWeight += vWeights[i];
	}

	Mean3f = Mean3f / TotalWeight;

	for (int t = 0; t < 5; t++)
	{
		float scale = 3 / (1 << t);// ������3sigma��0.2sigma

		float Thresh = (scale * vSigma) * (scale * vSigma);

		int nIterNum = 0;
		while (1)
		{
			Vec3f CurMean3f = Vec3f{ 0,0,0 };
			int GroupNum = 0;

			TotalWeight = 0.0f;
			for (int i = 0; i < vColors.size(); i++)
			{
				Vec3f Diff = (Vec3f)vColors[i] - Mean3f;

				if (Diff[0] * Diff[0] + Diff[1] * Diff[1] + Diff[2] * Diff[2] < Thresh)
				{
					CurMean3f += vColors[i] * vWeights[i];
					TotalWeight += vWeights[i];
					GroupNum++;
				}
			}

			if (GroupNum == 0)
			{
				break;
			}
			CurMean3f = CurMean3f / TotalWeight;

			Vec3f Diff = CurMean3f - Mean3f;

			if (Diff[0] * Diff[0] + Diff[1] * Diff[1] + Diff[2] * Diff[2] < 10)
			{
				break;
			}

			Mean3f = CurMean3f;

			nIterNum++;
			if (nIterNum > 10)
				break;
		}
	}
	return Mean3f;
}