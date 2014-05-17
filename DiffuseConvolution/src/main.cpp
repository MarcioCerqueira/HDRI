#include <opencv2/opencv.hpp>
#include "loadsave.h"

#define PI 3.14150

void Map2Direction(cv::Mat direction, int width, int height) {
	
	for(int y = 0; y < height; y++) {
		for(int x = 0; x < width; x++) {
			
			float nx = x/(float)width;
			float ny = y/(float)height;

			//float phi = PI * 2 * nx;
			float phi = PI * (nx * 2 - 1) - PI * 0.5; 
			float theta = PI * ny;
			float sinTheta = sinf(theta);
			int pixel = y * width + x;

			direction.ptr<float>()[pixel * 3 + 0] = cosf(phi) * sinTheta;
			direction.ptr<float>()[pixel * 3 + 1] = cosf(theta);
			direction.ptr<float>()[pixel * 3 + 2] = sinf(phi) * sinTheta;
			
		}
	}

}

void sinImage(cv::Mat map) {
	
	int width = map.cols;
	int height = map.rows;

	for(int y = 0; y < height; y++) {
	
		float ny = y/(float)height;
		float theta = PI * ny;
		float sinTheta = sinf(theta);

		for(int x = 0; x < width; x++) {
			
			int pixel = y * width + x;
			for(int ch = 0; ch < 3; ch++) {
				map.ptr<float>()[pixel * 3 + ch] *= sinTheta;
			}

		}

	}

}

void Direction2SH(float (*SH)[9], cv::Mat floatMap, cv::Mat directionMap, float y00, float y1x, float y2x, float y20, float y22) {

	std::vector<cv::Mat> floatMapPerCh;
	std::vector<cv::Mat> directionMapPerCh;

	cv::split(floatMap, floatMapPerCh);
	cv::split(directionMap, directionMapPerCh);

	for(int ch = 0; ch < 3; ch++) {
		
		cv::Scalar s0 = cv::mean(floatMapPerCh[ch] * y00);
		SH[ch][0] = s0(0);

		cv::Scalar s1 = cv::mean(floatMapPerCh[ch].mul(directionMapPerCh[1]) * y1x);
		SH[ch][1] = s1(0);

		cv::Scalar s2 = cv::mean(floatMapPerCh[ch].mul(directionMapPerCh[2]) * y1x);
		SH[ch][2] = s2(0);

		cv::Scalar s3 = cv::mean(floatMapPerCh[ch].mul(directionMapPerCh[0]) * y1x);
		SH[ch][3] = s3(0);

		cv::Scalar s4 = cv::mean(floatMapPerCh[ch].mul(directionMapPerCh[0].mul(directionMapPerCh[1])) * y2x);
		SH[ch][4] = s4(0);

		cv::Scalar s5 = cv::mean(floatMapPerCh[ch].mul(directionMapPerCh[1].mul(directionMapPerCh[2])) * y2x);
		SH[ch][5] = s5(0);

		cv::Scalar s6 = cv::mean(floatMapPerCh[ch].mul(directionMapPerCh[0].mul(directionMapPerCh[2])) * y2x);
		SH[ch][6] = s6(0);

		cv::Scalar s7 = cv::mean(floatMapPerCh[ch].mul(3 * directionMapPerCh[2].mul(directionMapPerCh[2]) - 1) * y20);
		SH[ch][7] = s7(0);

		cv::Scalar s8 = cv::mean(floatMapPerCh[ch].mul(directionMapPerCh[0].mul(directionMapPerCh[0]) - directionMapPerCh[1].mul(directionMapPerCh[1])) * y22);
		SH[ch][8] = s8(0);

	}

	//scaling
	for(int ch = 0; ch < 3; ch++)
		for(int s = 0; s < 9; s++)
			SH[ch][s] *= PI * PI * 2;

}

void evaluateSH(cv::Mat outputMap, float SH[][9], cv::Mat directionMap) {

	float c[5];
	c[0] = 0.429043;
	c[1] = 0.511664;
	c[2] = 0.743125;
	c[3] = 0.886227;
	c[4] = 0.247708;

	std::vector<cv::Mat> outputCh;
	std::vector<cv::Mat> directionCh;

	cv::split(outputMap, outputCh);
	cv::split(directionMap, directionCh);

	for(int ch = 0; ch < 3; ch++) {
		outputCh[ch] = c[0] * SH[ch][8] * (directionCh[0].mul(directionCh[0]) - directionCh[1].mul(directionCh[1])) +
			c[2] * SH[ch][6] * directionCh[2].mul(directionCh[2]) + 
			c[3] * SH[ch][0] - 
			c[4] * SH[ch][6] +
			2 * c[0] * (SH[ch][4] * (directionCh[0].mul(directionCh[1])) + SH[ch][7] * (directionCh[0].mul(directionCh[2])) + 
				SH[ch][5] * directionCh[1].mul(directionCh[2])) +
			2 * c[1] * (SH[ch][3] * directionCh[0] + SH[ch][1] * directionCh[1] + SH[ch][2] * directionCh[2]);
	}

	
	cv::merge(outputCh, outputMap);

}

int main(int argc, char **argv) {
	
	if(argc < 3) {
		std::cout << "DiffuseConvolution.exe input.hdr output.hdr SHScale" << std::endl;
		return 0;
	}

	cv::Mat latitudeLongitudeMap, floatMap, directionMap, outputMap;
	//latitudeLongitudeMap = cv::imread(argv[1]);
	latitudeLongitudeMap = cv::hdrImread(argv[1], CV_LOAD_IMAGE_ANYDEPTH | CV_LOAD_IMAGE_COLOR);
	directionMap = cv::Mat(latitudeLongitudeMap.rows, latitudeLongitudeMap.cols, CV_32FC3);
	outputMap = cv::Mat(latitudeLongitudeMap.rows, latitudeLongitudeMap.cols, CV_32FC3);

	float SH[3][9];
	
	//projection constants
	float y00 = 0.282095;
	float y1x = 0.488603;
	float y2x = 1.092548;
	float y20 = 0.315392;
	float y22 = 0.546274;

	Map2Direction(directionMap, latitudeLongitudeMap.cols, latitudeLongitudeMap.rows);
	sinImage(latitudeLongitudeMap);
	Direction2SH(SH, latitudeLongitudeMap, directionMap, y00, y1x, y2x, y20, y22);
	for(int ch = 0; ch < 3; ch++) for(int s = 0; s < 9; s++) SH[ch][s] *= atof(argv[3]);
	//for(int s = 0; s < 9; s++) for(int ch = 0; ch < 3; ch++) std::cout << SH[ch][s] << std::endl;
	//system("pause");

	evaluateSH(outputMap, SH, directionMap);
	
	std::vector<int> params;
	params.push_back(cv::HDR_RLE);
	cv::hdrImwrite(argv[2], outputMap, params);

	return 0;

}