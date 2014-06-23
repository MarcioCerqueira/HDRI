#include <opencv2/opencv.hpp>
#include <time.h>
#include "IO/loadsave.h"
#include "HDR/HDRImage.h"

int main(int argc, char **argv) {
	
	if(argc < 1) {
		std::cout << "SphericalHarmonics.exe input.hdr" << std::endl;
		return 0;
	}

	cv::Mat lightProbe, envMap, dirMap;
	HDRImage *hdrImage;

	lightProbe = cv::hdrImread(argv[1], CV_LOAD_IMAGE_ANYDEPTH | CV_LOAD_IMAGE_COLOR);
	envMap = lightProbe;
	dirMap = lightProbe;
	
	cv::resize(lightProbe, lightProbe, cv::Size(256, 256));
	cv::resize(envMap, envMap, cv::Size(256, 256));
	cv::resize(dirMap, dirMap, cv::Size(256, 256));

	//some precomputations
	hdrImage = new HDRImage(lightProbe.cols, lightProbe.rows);
	hdrImage->computeCoordinates();
	hdrImage->computeDomegaProduct();
	hdrImage->setScale(0.05);

	//per frame
	hdrImage->load(lightProbe.ptr<float>());
	hdrImage->computeSHCoeffs();
	hdrImage->computeSphericalMap();
	memcpy(envMap.ptr<float>(), hdrImage->getImage(), lightProbe.rows * lightProbe.cols * 3 * sizeof(float));
	
	hdrImage->computeDominantLightDirection();
	hdrImage->computeDominantLightColor();
	hdrImage->computeSphericalMap();
	memcpy(dirMap.ptr<float>(), hdrImage->getImage(), lightProbe.rows * lightProbe.cols * 3 * sizeof(float));
	
	cv::pow(lightProbe, 1.0f / 2.2, lightProbe);
	cv::pow(envMap, 1.0f / 2.2, envMap);
	cv::pow(dirMap, 1.0f / 2.2, dirMap);

	//float seconds = (float)(end - start) / CLOCKS_PER_SEC;
	//printf("%f s\n", seconds);
	
	while(cv::waitKey(33) != 27) {
		cv::imshow("Original Light Probe", lightProbe); 
		cv::imshow("3-Band Environment Map", envMap);
		cv::imshow("3-Band Directional Map", dirMap);
	}

	return 0;
	delete hdrImage;

}