#include "HDR/GPUHDRImage.h"

__device__ float sincKernel(float x) {               
  if (fabs(x) < 1.0e-4) return 1.0 ;
  else return(sin(x)/x) ;
}

__global__ void computeCoordinates(float* cartesianCoord, float* sphericalCoord, int width, int height) 
{

	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if(idx > width * height) return;

	float u, v, r;
	float midWidth = width/2;
	float midHeight = height/2;
	int pixel = idx;
	int x = pixel % width;
	int y = pixel / width;

	u = (x - midWidth)/midWidth;
	v = (y - midHeight)/midHeight;
	r = sqrtf(u * u + v * v);

	if(r > 1.0) {
			
		cartesianCoord[pixel * 3 + 0] = 0;
		cartesianCoord[pixel * 3 + 1] = 0;
		cartesianCoord[pixel * 3 + 2] = 0;
		sphericalCoord[pixel * 2 + 0] = 0;
		sphericalCoord[pixel * 2 + 1] = 0;
			
	} else {
			
		float phi = atan2(v, u);
		float theta = PI * r;

		if(theta != theta) theta = 0;
		if(phi != phi) phi = 0;

		sphericalCoord[pixel * 2 + 0] = theta;
		sphericalCoord[pixel * 2 + 1] = phi;
		cartesianCoord[pixel * 3 + 0] = sin(theta) * cos(phi);
		cartesianCoord[pixel * 3 + 1] = sin(theta) * sin(phi);
		cartesianCoord[pixel * 3 + 2] = cos(theta);
				
	}

}

void GPUComputeCoordinates2(float* cartesianCoord, float* sphericalCoord, int width, int height) 
{

	int blockSize, numBlocks, imageSize;
	imageSize = width * height;
	if(imageSize > 512)
		blockSize = 512;
	else
		blockSize = imageSize % 512;
	numBlocks = (imageSize / 512) + 1;

	computeCoordinates<<<numBlocks, blockSize>>>(cartesianCoord, sphericalCoord, width, height);
	cudaError_t error = cudaGetLastError();
	if(error != cudaSuccess)
		printf("HDRImage.cu - GPUComputeCoordinates2: %s\n", cudaGetErrorString(error));
	
	cudaThreadSynchronize();
}

__global__ void computeDomegaProduct(float *cartesianCoord, float* sphericalCoord, float *domegaProduct, int width, int height) 
{

	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if(idx > width * height) return;
	
	float theta, dx, dy, dz, c, domega;
	int pixel = idx;
	
	theta = sphericalCoord[pixel * 2 + 0]; 
	dx = cartesianCoord[pixel * 3 + 0];
	dy = cartesianCoord[pixel * 3 + 1];
	dz = cartesianCoord[pixel * 3 + 2];
	domega = (2*PI/(float)width)*(2*PI/(float)width)*sincKernel(theta);
	c = 0.282095;
	domegaProduct[pixel * 9 + 0] = c * domega;
	c = 0.488603;
	domegaProduct[pixel * 9 + 1] = c * dy * domega;
	domegaProduct[pixel * 9 + 2] = c * dz * domega;
	domegaProduct[pixel * 9 + 3] = c * dx * domega;
	c = 1.092548;
	domegaProduct[pixel * 9 + 4] = c * dx * dy * domega;
	domegaProduct[pixel * 9 + 5] = c * dy * dz * domega;
	domegaProduct[pixel * 9 + 7] = c * dx * dz * domega;
	c = 0.315392;
	domegaProduct[pixel * 9 + 6] = c * (3 * dz * dz - 1) * domega;
	c = 0.546274;
	domegaProduct[pixel * 9 + 8] = c * (dx * dx - dy * dy) * domega;

}

void GPUComputeDomegaProduct2(float *cartesianCoord, float* sphericalCoord, float *domegaProduct, int width, int height)
{

	int blockSize, numBlocks, imageSize;
	imageSize = width * height;
	if(imageSize > 512)
		blockSize = 512;
	else
		blockSize = imageSize % 512;
	numBlocks = (imageSize / 512) + 1;

	computeDomegaProduct<<<numBlocks, blockSize>>>(cartesianCoord, sphericalCoord, domegaProduct, width, height);
	cudaError_t error = cudaGetLastError();
	if(error != cudaSuccess)
		printf("HDRImage.cu - GPUComputeDomegaProduct2: %s\n", cudaGetErrorString(error));
	
	cudaThreadSynchronize();
}


__global__ void computeSHCoeffs(float *image, float *domegaProduct, float *SHCoeffs, int width, int height, int i) 
{

	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if(idx > width * height) return;

	int pixel = idx;
	float u, v, r;
	float midWidth = width/2;
	float midHeight = height/2;
	int x = pixel % width;
	int y = pixel / width;

	u = (x - midWidth)/midWidth;
	v = (y - midHeight)/midHeight;
	r = sqrtf(u * u + v * v);
	if(r > 1.0) return;

	/*
	for(int ch = 0; ch < 3; ch++) {
		float imageValue = image[pixel * 3 + ch];
		atomicAdd(&SHCoeffs[0 * 3 + ch], imageValue * domegaProduct[pixel * 9 + 0]);
		atomicAdd(&SHCoeffs[1 * 3 + ch], imageValue * domegaProduct[pixel * 9 + 1]);
		atomicAdd(&SHCoeffs[2 * 3 + ch], imageValue * domegaProduct[pixel * 9 + 2]);
		atomicAdd(&SHCoeffs[3 * 3 + ch], imageValue * domegaProduct[pixel * 9 + 3]);
		atomicAdd(&SHCoeffs[4 * 3 + ch], imageValue * domegaProduct[pixel * 9 + 4]);
		atomicAdd(&SHCoeffs[5 * 3 + ch], imageValue * domegaProduct[pixel * 9 + 5]);
		atomicAdd(&SHCoeffs[7 * 3 + ch], imageValue * domegaProduct[pixel * 9 + 7]);
		atomicAdd(&SHCoeffs[6 * 3 + ch], imageValue * domegaProduct[pixel * 9 + 6]);
		atomicAdd(&SHCoeffs[8 * 3 + ch], imageValue * domegaProduct[pixel * 9 + 8]);
	}
	*/

	int sh = i / 3;
	int ch = i % 3;
	SHCoeffs[pixel] = image[pixel * 3 + ch] * domegaProduct[pixel * 9 + sh];
	
}

__global__ void scaleSHCoeffs(float *SHCoeffs, float scale) 
{

	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	SHCoeffs[idx] *= scale;

}

__global__ void setZeroSHCoeffs(float *SHCoeffs) 
{

	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	SHCoeffs[idx] = 0;

}

void GPUComputeSHCoeffs2(float *image, float *domegaProduct, float* SHCoeffs, float *auxSHCoeffs, float scale, int width, int height)
{

	int blockSize, numBlocks, imageSize;
	imageSize = width * height;
	if(imageSize > 512)
		blockSize = 512;
	else
		blockSize = imageSize % 512;
	numBlocks = (imageSize / 512) + 1;

	cublasHandle_t handle;
	cublasCreate(&handle);
	cublasStatus_t stat;
	
	setZeroSHCoeffs<<<3, 9>>>(SHCoeffs);
	
	float sum[27];
	for(int i = 0; i < 27; i++) {
		setZeroSHCoeffs<<<numBlocks, blockSize>>>(auxSHCoeffs);
		computeSHCoeffs<<<numBlocks, blockSize>>>(image, domegaProduct, auxSHCoeffs, width, height, i);
		stat = cublasSasum(handle, width * height, auxSHCoeffs, 1, &sum[i]);
	}
	
	cublasDestroy(handle);
	cudaMemcpy(SHCoeffs, sum, 27 * sizeof(float), cudaMemcpyHostToDevice);
	scaleSHCoeffs<<<3, 9>>>(SHCoeffs, scale);
	
	/*
	computeSHCoeffs<<<numBlocks, blockSize>>>(image, domegaProduct, SHCoeffs, width, height, 0);
	scaleSHCoeffs<<<3, 9>>>(SHCoeffs, scale);
	*/
	cudaError_t error = cudaGetLastError();
	if(error != cudaSuccess)
		printf("HDRImage.cu - GPUComputeSHCoeffs2: %s\n", cudaGetErrorString(error));
	
	cudaThreadSynchronize();

}

__global__ void computeSphericalMap(float *cartesianCoord, float *SHCoeffs, float *image, int width, int height)
{

	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if(idx > width * height) return;

	int pixel = idx;
	
	float c[5];
	c[0] = 0.429043;
	c[1] = 0.511664;
	c[2] = 0.743125;
	c[3] = 0.886227;
	c[4] = 0.247708;

	float dx, dy, dz;
		
	int x = pixel % width;
	int y = pixel / width;

	dx = cartesianCoord[pixel * 3 + 0];
	dy = cartesianCoord[pixel * 3 + 1];
	dz = cartesianCoord[pixel * 3 + 2];
	
	if(dx == 0 && dy == 0 && dz == 0) {

		for(int ch = 0; ch < 3; ch++)
			image[pixel * 3 + ch] = 0;
		
	} else {

		for(int ch = 0; ch < 3; ch++) {

			image[pixel * 3 + ch] = c[0] * SHCoeffs[8 * 3 + ch] * (dx * dx - dy * dy) + 
				c[2] * SHCoeffs[6 * 3 + ch] * dz * dz +
				c[3] * SHCoeffs[0 * 3 + ch] -
				c[4] * SHCoeffs[6 * 3 + ch] +
				2 * c[0] * (SHCoeffs[4 * 3 + ch] * dx * dy + SHCoeffs[7 * 3 + ch] * dx * dz + SHCoeffs[5 * 3 + ch] * dy * dz) +
				2 * c[1] * (SHCoeffs[3 * 3 + ch] * dx + SHCoeffs[1 * 3 + ch] * dy + SHCoeffs[2 * 3 + ch] * dz);
			if(image[pixel * 3 + ch] < 0) image[pixel * 3 + ch] = 0;

		}

	}

}

void GPUComputeSphericalMap2(float *cartesianCoord, float *SHCoeffs, float *image, int width, int height)
{

	int blockSize, numBlocks, imageSize;
	imageSize = width * height;
	if(imageSize > 512)
		blockSize = 512;
	else
		blockSize = imageSize % 512;
	numBlocks = (imageSize / 512) + 1;

	computeSphericalMap<<<numBlocks, blockSize>>>(cartesianCoord, SHCoeffs, image, width, height);
	
	cudaError_t error = cudaGetLastError();
	if(error != cudaSuccess)
		printf("HDRImage.cu - GPUComputeSHCoeffs2: %s\n", cudaGetErrorString(error));
	
	cudaThreadSynchronize();

}
