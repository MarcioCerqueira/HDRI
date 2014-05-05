#include <opencv2/opencv.hpp>
#include "loadsave.h"

#define PI 3.14150
#define sign(x) x/(abs(x)+(x==0))

void bilinearInterpolation(float nx, float ny, float fx, float fy, int width, int height, int pixel, cv::Mat dst, cv::Mat src) {
	
	//float fx = nx * width;
	//float fy = ny * height;
	int ix0 = int(fx);
	int iy0 = int(fy);
	float tx = fx - ix0;
	float ty = fy - iy0;
	int ix1 = std::min<int>(ix0 + 1, width - 1);
	int iy1 = std::min<int>(iy0 + 1, height - 1);
	int pixel1 = iy0 * width + ix0;
	int pixel2 = iy0 * width + ix1;
	int pixel3 = iy1 * width + ix0;
	int pixel4 = iy1 * width + ix0;
	
	if(src.type() == CV_8UC3) {
	
		unsigned char pix1[3], pix2[3], pix3[3], pix4[3];
		unsigned char* srcData = src.ptr<unsigned char>();
		unsigned char* dstData = dst.ptr<unsigned char>();

		for(int ch = 0; ch < 3; ch++) {

			pix1[ch] = srcData[pixel1 * 3 + ch];
			pix2[ch] = srcData[pixel2 * 3 + ch];
			pix3[ch] = srcData[pixel3 * 3 + ch];
			pix4[ch] = srcData[pixel4 * 3 + ch];
	
			dstData[pixel * 3 + ch] = pix1[ch] * (1 - tx) * (1 - ty) + pix2[ch] * tx * (1 - ty) + pix3[ch] * (1 - tx) * ty + pix4[ch] * tx * ty;
	
		}

	} else {

		float pix1[3], pix2[3], pix3[3], pix4[3];
		float* srcData = src.ptr<float>();
		float* dstData = dst.ptr<float>();

		for(int ch = 0; ch < 3; ch++) {

			pix1[ch] = srcData[pixel1 * 3 + ch];
			pix2[ch] = srcData[pixel2 * 3 + ch];
			pix3[ch] = srcData[pixel3 * 3 + ch];
			pix4[ch] = srcData[pixel4 * 3 + ch];
	
			dstData[pixel * 3 + ch] = pix1[ch] * (1 - tx) * (1 - ty) + pix2[ch] * tx * (1 - ty) + pix3[ch] * (1 - tx) * ty + pix4[ch] * tx * ty;
	
		}

	}
}

void LL2Direction(float *direction, int width, int height) {
	
	for(int y = 0; y < height; y++) {
		for(int x = 0; x < width; x++) {
			
			float nx = x/(float)width;
			float ny = y/(float)height;

			float phi = PI * (nx * 2 - 1) - PI * 0.5; 
			float theta = ny * PI;
			int pixel = y * width + x;

			direction[pixel * 3 + 0] = cosf(phi) * sinf(theta);
			direction[pixel * 3 + 1] = cosf(theta);
			direction[pixel * 3 + 2] = sinf(phi) * sinf(theta);
			
		}
	}

}

void Ang2Direction(float *direction, int width, int height) {
	
	for(int y = 0; y < height; y++) {
		for(int x = 0; x < width; x++) {
			
			float nx = x/(float)width;
			float ny = y/(float)height;

			float theta = atan2(-2 * ny + 1, 2 * nx - 1);
			float phi = PI * sqrt(pow(2 * nx - 1, 2) + pow(2 * ny - 1, 2));
			int pixel = y * width + x;

			direction[pixel * 3 + 0] = cosf(theta) * sin(phi);
			direction[pixel * 3 + 1] = sin(theta) * sin(phi);
			direction[pixel * 3 + 2] = -cos(phi);

		}
	}

}

void Cube2Direction(float *direction, int width, int height) {
	
	int tile = width/3;

	float *tileDir = (float*)malloc(tile * tile * 3 * sizeof(float));

	for(int y = 0; y < tile; y++) {
		for(int x = 0; x < tile; x++) {
			
			int pixel = y * tile + x;
			tileDir[pixel * 3 + 0] = x/(float)tile * 2 - 1;
			tileDir[pixel * 3 + 1] = 1;
			tileDir[pixel * 3 + 2] = y/(float)tile * 2 - 1;
			float n = sqrtf(tileDir[pixel * 3 + 0] * tileDir[pixel * 3 + 0] + tileDir[pixel * 3 + 1] * tileDir[pixel * 3 + 1] + tileDir[pixel * 3 + 2] * tileDir[pixel * 3 + 2]);
			tileDir[pixel * 3 + 0] /= n;
			tileDir[pixel * 3 + 1] /= n;
			tileDir[pixel * 3 + 2] /= n;

		}
	}

	int C1[] = {0, 2 * tile, tile, 3 * tile, tile, tile};
	int C2[] = {tile, 3 * tile, 2 * tile, 4 * tile, 2 * tile, 2 * tile};
	int C3[] = {tile, tile, tile, tile, 0, 2 * tile};
	int C4[] = {2 * tile, 2 * tile, 2 * tile, 2 * tile, tile, 3 * tile};
	int X[] = {1, 1, 1, 1, -2, 2};
	int Y[] = {2, -2, -3, 3, -3, -3};
	int Z[] = {-3, 3, -2, 2, -1, 1};

	int count = 0;
	for(int i = 0; i < 6; i++) {
		for(int x = C3[i]; x < C4[i]; x++) {
			for(int y = C1[i]; y < C2[i]; y++) {
				
				int pixel = y * width + x;
				int tilePixel = (y-C1[i]) * tile + (x-C3[i]);
				direction[pixel * 3 + 0] = sign(X[i]) * tileDir[tilePixel * 3 + abs(X[i])-1];
				direction[pixel * 3 + 1] = sign(Y[i]) * tileDir[tilePixel * 3 + abs(Y[i])-1];
				direction[pixel * 3 + 2] = sign(Z[i]) * tileDir[tilePixel * 3 + abs(Z[i])-1];
				count++;

			}
		}
	}

	delete [] tileDir;

}

void Direction2LL(cv::Mat dst, cv::Mat src, float *direction, int width, int height, int oldWidth, int oldHeight) {

	for(int pixel = 0; pixel < width * height; pixel++) {

		float nx = 1.0 + atan2(direction[pixel * 3 + 0], -direction[pixel * 3 + 2]) / PI;
		float ny = acosf(direction[pixel * 3 + 1]) / PI;
		int x = nx * oldWidth/2;
		int y = ny * oldHeight; 
		int oldPixel = y * oldWidth + x;

		if(oldPixel >= 0 && oldPixel < oldWidth * oldHeight)
			bilinearInterpolation(nx, ny, x, y, oldWidth, oldHeight, pixel, dst, src);

	}

}

void Direction2Ang(cv::Mat dst, cv::Mat src, float *direction, int width, int height, int oldWidth, int oldHeight) {
	
	for(int pixel = 0; pixel < width * height; pixel++) {

		float r = acos(-direction[pixel * 3 + 2])/(PI * 2 * sqrt(powf(direction[pixel * 3 + 0], 2) + powf(direction[pixel * 3 + 1], 2)));
		
		float nx = 0.5 + r * direction[pixel * 3 + 0];
		float ny = 0.5 - r * direction[pixel * 3 + 1];
		int x = nx * oldWidth;
		int y = ny * oldHeight;
		int oldPixel = y * oldWidth + x;

		if(oldPixel >= 0 && oldPixel < oldWidth * oldHeight)
			bilinearInterpolation(nx, ny, x, y, oldWidth, oldHeight, pixel, dst, src);

	}

}

void Direction2Cube(cv::Mat dst, cv::Mat src, float *direction, int width, int height, int oldWidth, int oldHeight) {
	
	int Mul[] = {1, 1, 1, -1, -1, -1};
	int T[] = {1, 0, 2, 1, 0, 2};
	int A[] = {0, 2, 0, 0, 2, 0};
	int B[] = {2, 1, 1, 2, 1, 1};
	float X1_1[] = {1.5, 2.5, 1.5, 1.5, 0.5, 1.5};
	float X1_2[] = {0.5, 0.5, 0.5, -0.5, 0.5, -0.5};
	float Y1_1[] = {2.5, 1.5, 3.5, 0.5, 1.5, 1.5};
	float Y1_2[] = {0.5, 0.5, -0.5, 0.5, -0.5, -0.5};

	for(int i = 0; i < 6; i++) {
		
		int t = T[i];
		int a = A[i];
		int b = B[i];

		for(int pixel = 0; pixel < width * height; pixel++) {
			
			if(Mul[i] * direction[pixel * 3 + t] > 0 && Mul[i] * direction[pixel * 3 + t] >= abs(direction[pixel * 3 + a]) &&
				Mul[i] * direction[pixel * 3 + t] >= abs(direction[pixel * 3 + b]) && direction[pixel * 3 + t] > 0) {
			
				float nx = X1_1[i] + X1_2[i] * direction[pixel * 3 + a]/direction[pixel * 3 + t];
				float ny = Y1_1[i] + Y1_2[i] * direction[pixel * 3 + b]/direction[pixel * 3 + t];
				int x = nx * oldWidth;
				int y = ny * oldHeight;
				int oldPixel = y * oldWidth + x;

				if(oldPixel >= 0 && oldPixel < oldWidth * oldHeight)
					bilinearInterpolation(nx, ny, x, y, oldWidth, oldHeight, pixel, dst, src);

			}

		}


	}

}

void AngMask(cv::Mat map, int width, int height) {

	if(map.type() == CV_8UC3) {

		unsigned char* mapData = map.ptr<unsigned char>();
		for(int y = 0; y < height; y++) {
			for(int x = 0; x < width; x++) {
				float nx = x/(float)(width) * 2 - 1;
				float ny = y/(float)(height) * 2 - 1;
				float r = sqrtf(nx * nx + ny * ny);
				int pixel = y * width + x;
				if(r > 1) {
					mapData[pixel * 3 + 0] = 0;
					mapData[pixel * 3 + 1] = 0;
					mapData[pixel * 3 + 2] = 0;
				}
			}
		}

	} else {

		float* mapData = map.ptr<float>();
		for(int y = 0; y < height; y++) {
			for(int x = 0; x < width; x++) {
				float nx = x/(float)(width) * 2 - 1;
				float ny = y/(float)(height) * 2 - 1;
				float r = sqrtf(nx * nx + ny * ny);
				int pixel = y * width + x;
				if(r > 1) {
					mapData[pixel * 3 + 0] = 0;
					mapData[pixel * 3 + 1] = 0;
					mapData[pixel * 3 + 2] = 0;
				}
			}
		}

	}

}

void CubeMask(cv::Mat map, int width, int height) {
	
	int tile = width/3;

	int *mask = (int*)malloc(width * height * sizeof(int));

	for(int pixel = 0; pixel < width * height; pixel++)
		mask[pixel] = 0;

	for(int y = tile; y < 2 * tile; y++)
		for(int x = 0; x < 3 * tile; x++)
			mask[y * width + x] = 1;

	for(int x = tile; x < 2 * tile; x++) {
		for(int y = 0; y < tile; y++)
			mask[y * width + x] = 1;
		for(int y = 2 * tile; y < 4 * tile; y++)
			mask[y * width + x] = 1;
	}

	if(map.type() == CV_8UC3) {
	
		unsigned char *mapData = map.ptr<unsigned char>();
		for(int pixel = 0; pixel < width * height; pixel++) {
			mapData[pixel * 3 + 0] *= mask[pixel];
			mapData[pixel * 3 + 1] *= mask[pixel];
			mapData[pixel * 3 + 2] *= mask[pixel];
		}
	
	} else {

		float *mapData = map.ptr<float>();
		for(int pixel = 0; pixel < width * height; pixel++) {
			mapData[pixel * 3 + 0] *= mask[pixel];
			mapData[pixel * 3 + 1] *= mask[pixel];
			mapData[pixel * 3 + 2] *= mask[pixel];
		}
	
	}

	delete [] mask;

}

int main(int argc, char **argv) {
	
	if(argc < 6) {
		std::cout << "ChangeMapping.exe input.png output.png inputFormat outputFormat imageFormat" << std::endl;
		std::cout << "Format:" << std::endl;
		std::cout << "\t LL: Latitude-Longitude Map" << std::endl;
		std::cout << "\t Ang: Angular Map" << std::endl;
		std::cout << "\t Cube: Cube Map" << std::endl; 
		std::cout << "ImageFormat:" << std::endl;
		std::cout << "ldr or hdr" << std::endl;
		return 0;
	}

	cv::Mat environmentMap;
	if(!strcmp(argv[5], "ldr"))
		environmentMap = cv::imread(argv[1]);
	else
		environmentMap = cv::hdrImread(argv[1], CV_LOAD_IMAGE_ANYDEPTH | CV_LOAD_IMAGE_COLOR);

	int newWidth, newHeight;
	
	if(!strcmp(argv[3], "LL")) {
		newWidth = environmentMap.cols / 2;
		newHeight = environmentMap.rows;
	}
	if(!strcmp(argv[3], "Ang")) {
		newWidth = environmentMap.cols;
		newHeight = environmentMap.rows;
	}
	if(!strcmp(argv[3], "Cube")) {
		newWidth = environmentMap.cols * 3;
		newHeight = environmentMap.rows * 4;
	}

	if(!strcmp(argv[4], "LL"))
		newWidth *= 2;
	if(!strcmp(argv[4], "Cube")) {
		newWidth *= 3;
		newHeight *= 4;
	}

	float *direction = (float*)malloc(newWidth * newHeight * 3 * sizeof(float));

	if(!strcmp(argv[4], "LL"))
		LL2Direction(direction, newWidth, newHeight);
	if(!strcmp(argv[4], "Ang"))
		Ang2Direction(direction, newWidth, newHeight);
	if(!strcmp(argv[4], "Cube"))
		Cube2Direction(direction, newWidth, newHeight);

	int type;
	if(!strcmp(argv[5], "ldr"))
		type = CV_8UC3;
	else
		type = CV_32FC3;

	cv::Mat outputMap(newHeight, newWidth, type);
	outputMap = cv::Mat::zeros(outputMap.size(), outputMap.type());
	
	if(!strcmp(argv[3], "LL"))
		Direction2LL(outputMap, environmentMap, direction, newWidth, newHeight, environmentMap.cols, environmentMap.rows);
	if(!strcmp(argv[3], "Ang"))
		Direction2Ang(outputMap, environmentMap, direction, newWidth, newHeight, environmentMap.cols, environmentMap.rows);
	if(!strcmp(argv[3], "Cube"))
		Direction2Cube(outputMap, environmentMap, direction, newWidth, newHeight, environmentMap.cols, environmentMap.rows);

	if(!strcmp(argv[4], "Ang"))
		AngMask(outputMap, newWidth, newHeight);
	if(!strcmp(argv[4], "Cube"))
		CubeMask(outputMap, newWidth, newHeight);

	std::vector<int> params;
	params.push_back(cv::HDR_RLE);
	
	if(!strcmp(argv[5], "ldr"))
		cv::imwrite(argv[2], outputMap);
	else
		cv::hdrImwrite(argv[2], outputMap, params);
	
	delete [] direction;
	
	return 0;

}