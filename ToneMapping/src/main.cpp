#include <opencv2/opencv.hpp>
#include "loadsave.h"

float gamma = 1;
float intensity = 0;
float sigmaColor = 0;
float sigmaLight = 1;

int intensitySlider;
int gammaSlider;
int sigmaColorSlider;
int sigmaLightSlider;

cv::Mat hdrImage;
cv::Mat result;

void updateToneMappedImage(int, void*) {

	double min, max;
	double logMin, logMax;

	gamma = 0.3 + (float)(gammaSlider)/10.0;
	intensity = intensitySlider - 8;
	sigmaColor = (float)(sigmaColorSlider)/10.0;
	sigmaLight = (float)(sigmaLightSlider)/10.0;

	cv::Mat grayImage, logImage;
	cv::cvtColor(hdrImage, grayImage, CV_RGB2GRAY);
	cv::log(grayImage, logImage);

	float logMean = cv::sum(logImage)[0] / logImage.total();
	cv::minMaxLoc(logImage, &logMin, &logMax);
	
	double key = (logMax - logMean) / (logMax - logMin);
	float mapKey = 0.3 + 0.7 * pow(key, 1.4);

	intensity = expf(-intensity);
	cv::Scalar chanMean = cv::mean(hdrImage);
	float grayMean = cv::mean(grayImage)[0];

	std::vector<cv::Mat> channels(3);
    cv::split(hdrImage, channels);

    for(int i = 0; i < 3; i++) {
        float global = sigmaColor * static_cast<float>(chanMean[i]) + (1.0f - sigmaColor) * grayMean;
        cv::Mat adapt = sigmaColor * channels[i] + (1.0f - sigmaColor) * grayImage;
        adapt = sigmaLight * adapt + (1.0f - sigmaLight) * global;
        pow(intensity * adapt, mapKey, adapt);
        channels[i] = channels[i].mul(1.0f / (adapt + channels[i]));
    }

    cv::merge(channels, result);

	cv::minMaxLoc(result, &min, &max);
	if(max - min > 1e-5) result = (result - min) / (max - min);
	cv::pow(result, 1.0f / gamma, result);
	
	cv::imshow("Reinhard's Tone Mapping Algorithm", result);

}

int main(int argc, char **argv) {
	
	if(argc < 3) {
		std::cout << "ToneMapping.exe file.hdr outputImage" << std::endl;
		return 0;
	}

	double min, max;
	hdrImage = cv::hdrImread(argv[1], CV_LOAD_IMAGE_ANYDEPTH | CV_LOAD_IMAGE_COLOR);
	cv::minMaxLoc(hdrImage, &min, &max);
	if(max - min > 1e-5) hdrImage = (hdrImage - min) / (max - min);
	cv::pow(hdrImage, 1.0f / gamma, hdrImage);

	updateToneMappedImage(0, 0);
	
	cv::namedWindow("Reinhard's Tone Mapping Algorithm");
	cv::createTrackbar("Gamma", "Reinhard's Tone Mapping Algorithm", &gammaSlider, 7, updateToneMappedImage);
	cv::createTrackbar("Intensity", "Reinhard's Tone Mapping Algorithm", &intensitySlider, 16, updateToneMappedImage);
	cv::createTrackbar("Color.adpt", "Reinhard's Tone Mapping Algorithm", &sigmaColorSlider, 10, updateToneMappedImage);
	cv::createTrackbar("Light.adpt", "Reinhard's Tone Mapping Algorithm", &sigmaLightSlider, 10, updateToneMappedImage);

	cv::waitKey(0);

	result.convertTo(result, CV_8U, 255.0);
	cv::imwrite(argv[2], result);
    
	return 0;
}