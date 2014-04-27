#include <opencv2/opencv.hpp>
#include "loadsave.h"
#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/SVD>
#include <vector>
#include <fstream>
#include <string>

typedef struct Point
{
	int x;
	int y;
};

bool isValid(IplImage* image, int pixel) {
	
	int r, g, b;
	r = (unsigned char)image->imageData[pixel * 3 + 0];
	g = (unsigned char)image->imageData[pixel * 3 + 1];
	b = (unsigned char)image->imageData[pixel * 3 + 2];
	if(r == 0 && g == 0 && b > 250)
		return false;
	else 
		return true;
					
}

void loadSequence(std::vector<IplImage*>& imageList, std::vector<float>& exposureTimeList, char *config) {
	
	std::fstream file(config);
	std::string line;
	bool inverseExposureTime;

	if(file.is_open()) {
		
		std::getline(file, line);
		std::getline(file, line);
		std::cout << "Number of images: " << atoi(line.c_str()) << std::endl;
		imageList.resize(atoi(line.c_str()));
		exposureTimeList.resize(atoi(line.c_str()));

		std::getline(file, line);
		std::getline(file, line);
		std::cout << "Inverse exposure time? " << atoi(line.c_str()) << std::endl;
		inverseExposureTime = atoi(line.c_str());

		std::getline(file, line);
		for(int image = 0; image < imageList.size(); image++) {
			
			std::getline(file, line);
			imageList[image] = cvLoadImage(line.c_str());
			cvCvtColor(imageList[image], imageList[image], CV_BGR2RGB);
			std::cout << "Image: " << line.c_str() << " loaded" << std::endl;
			std::getline(file, line);
			exposureTimeList[image] = atof(line.c_str());
			if(inverseExposureTime) exposureTimeList[image] = 1.f / exposureTimeList[image];

		}

	}

}

void computeWeightList(Eigen::VectorXf& weightList) {

	weightList.resize(256);
	for(int value = 0; value < weightList.size(); value++) {	
		if(value <= weightList.size()/2) weightList[value] = value;
		else weightList[value] = 256 - value;
	}
	
}

void computeSamplesList(std::vector<Point>& samplesList, int width, int height, int numberOfSamples) {

	int xPoints = sqrtf(numberOfSamples * width / height);
	int yPoints = numberOfSamples / xPoints;
	int xStep = width / xPoints;
	int yStep = height / yPoints;

	int k = 0;
	for(int i = 0, x = xStep / 2; i < xPoints; i++, x += xStep) {
		for(int j = 0, y = yStep / 2; j < yPoints; j++, y += yStep) {
			if(x > width * 0.1 && x < (width - width * 0.1) && y > (height * 0.1) && y < (height - height * 0.1)) {
				Point p;
				p.x = x;
				p.y = y;
				samplesList.push_back(p);
				k++;
			}
		}
	}

}

void computeCameraResponseFunction(std::vector<IplImage*>& imageList, std::vector<float>& exposureTimeList, Eigen::VectorXf& weightList, 
	Eigen::MatrixXf& responseList) {
	
	std::vector<Point> samplesList;

	int n = 256;
	float lambda = 10;
	int imageSize = imageList[0]->width * imageList[0]->height;
	int numberOfImages = imageList.size();
	int numberOfSamples = (n / numberOfImages) * 4;

	responseList.resize(n, 3);
	
	computeSamplesList(samplesList, imageList[0]->width, imageList[0]->height, numberOfSamples);
	numberOfSamples = samplesList.size();

	for(int ch = 0; ch < 3; ch++) {

		Eigen::MatrixXf A(numberOfSamples * numberOfImages + n + 1, n + numberOfSamples);
		Eigen::VectorXf b(A.rows(), 1);

		A.setZero();
		b.setZero();

		//Include the data-fitting equations
		int k = 0;
		int pixel;
		int val;
		for(int i = 0; i < numberOfSamples; i++) {
			for(int j = 0; j < numberOfImages; j++) {
				pixel = samplesList[i].y * imageList[0]->width + samplesList[i].x;
				val = (unsigned char)imageList[j]->imageData[pixel * 3 + ch];
				A(k, val) = weightList[val];
				A(k, n + i) = -weightList[val];
				b(k, 0) = weightList[val] * logf(exposureTimeList[j]);
				k++;
			}
		}

		//Fix the curve by setting its middle value to 0
		A(k, 128) = 1;
		k++;

		//Include the smoothness equations
		for(int i = 0; i < n - 2; i++) {
			A(k, i) = lambda * weightList[i + 1];
			A(k, i+1) = -2 * lambda * weightList[i + 1];
			A(k, i+2) = lambda * weightList[i + 1];
			k++;
		}

		Eigen::VectorXf x = A.jacobiSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(b);
		for(int i = 0; i < responseList.rows(); i++)
			responseList(i, ch) = expf(x(i));

	}

}

void buildHDRImage(std::vector<IplImage*>& imageList, std::vector<float>& exposureTimeList, Eigen::VectorXf& weightList, Eigen::MatrixXf& responseList, 
	Eigen::MatrixXf& hdrImage) {
	
	int imageSize = imageList[0]->width * imageList[0]->height;
	int numberOfImages = imageList.size();
	hdrImage.resize(imageSize, 3);
	hdrImage.setZero();
	float weightSum;

	for(int ch = 0; ch < 3; ch++) {
		for(int y = 0; y < imageList[0]->height; y++) {	
			for(int x = 0; x < imageList[0]->width; x++) {

				int pixel = y * imageList[0]->width + x;
				weightSum = 0;
				
				for(int image = 0; image < numberOfImages; image++) {
				
					if(isValid(imageList[image], pixel)) {
						
						int val = (unsigned char)imageList[image]->imageData[pixel * 3 + ch];
						hdrImage(pixel, ch) += weightList(val) * (logf(responseList(val, ch)) - logf(exposureTimeList[image]));
						weightSum += weightList(val);
					
					}
				
				}
				
				hdrImage(pixel, ch) /= weightSum;
				hdrImage(pixel, ch) = expf(hdrImage(pixel, ch));
			
			}
		}
	}
	
}

void saveHDRImage(Eigen::MatrixXf& hdrImage, int width, int height, char *fileName) {
	
	cv::Mat hdr(height, width, CV_32FC3);
	float *hdrData = hdr.ptr<float>();
	for(int pixel = 0; pixel < width * height; pixel++) {
		hdrData[pixel * 3 + 0] = hdrImage(pixel, 2);
		hdrData[pixel * 3 + 1] = hdrImage(pixel, 1);
		hdrData[pixel * 3 + 2] = hdrImage(pixel, 0);
	}

	std::vector<int> params;
	params.push_back(cv::HDR_RLE);
	cv::hdrImwrite(fileName, hdr, params);

}

int main(int argc, char **argv) {

	std::vector<IplImage*> imageList;
	std::vector<float> exposureTimeList;
	Eigen::MatrixXf responseList;
	Eigen::VectorXf weightList;
	Eigen::MatrixXf hdrImage;

	if(argc < 2) {
		std::cout << "LDR2HDR.exe config.txt output.hdr" << std::endl;
		return 0;
	} 

	loadSequence(imageList, exposureTimeList, argv[1]);	
	computeWeightList(weightList);
	computeCameraResponseFunction(imageList, exposureTimeList, weightList, responseList);
	buildHDRImage(imageList, exposureTimeList, weightList, responseList, hdrImage);
	saveHDRImage(hdrImage, imageList[0]->width, imageList[0]->height, argv[2]);

	return 0;

}