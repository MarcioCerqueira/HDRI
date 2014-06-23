#ifndef HDRIMAGE_H
#define HDRIMAGE_H

#include <malloc.h>
#include <math.h>
#include <stdio.h>
#include <memory.h>
#include <cuda_runtime.h>
#include "HDR\GPUHDRImage.h"
#include "HDR\SH.h"

class HDRImage
{
public:
	HDRImage(int width, int height);
	~HDRImage();
	void computeCoordinates();
	void computeDomegaProduct();
	void computeSHCoeffs();
	void computeSphericalMap();
	void computeDominantLightDirection();
	void computeDominantLightColor();
	void load(float *image);
	float* getImage() { return image; }
	void setScale(float scale) { this->scale = scale; }

	void GPUComputeCoordinates();
	void GPUComputeDomegaProduct();
	void GPUComputeSHCoeffs();
	void GPUComputeSphericalMap();
	void GPULoad(float *image);
	

private:
	float *image;
	float *cartesianCoord;
	float *sphericalCoord;
	float *domegaProduct;
	float SHCoeffs[9][3];
	float dominantLightDirection[3];
	float dominantLightColor[3];
	int width;
	int height;
	float scale;

	float *deviceImage;
	float *deviceCartesianCoord;
	float *deviceSphericalCoord;
	float *deviceDomegaProduct;
	float *deviceSHCoeffs;
	float *deviceAuxSHCoeffs;
};

#endif