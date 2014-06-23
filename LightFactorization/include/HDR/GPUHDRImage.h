#ifndef GPUHDRIMAGE_H
#define GPUHDRIMAGE_H

#include <cuda.h>
#include <cublas_v2.h>
#include <stdio.h>

#define PI 3.141593

void GPUComputeCoordinates2(float* cartesianCoord, float* sphericalCoord, int width, int height);
void GPUComputeDomegaProduct2(float *cartesianCoord, float* sphericalCoord, float *domegaProduct, int width, int height);
void GPUComputeSHCoeffs2(float *image, float *domegaProduct, float *SHCoeffs, float *auxSHCoeffs, float scale, int width, int height);
void GPUComputeSphericalMap2(float *cartesianCoord, float *SHCoeffs, float *image, int width, int height);

#endif