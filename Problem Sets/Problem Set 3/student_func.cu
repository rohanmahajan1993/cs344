#include "reference_calc.cpp"
#include "utils.h"

void your_histogram_and_prefixsum(const float* const d_logLuminance,
                                  unsigned int* const d_cdf,
                                  float &min_logLum,
                                  float &max_logLum,
                                  const size_t numRows,
                                  const size_t numCols,
                                  const size_t numBins)
{
  //TODO
  /*Here are the steps you need to implement
    1) find the minimum and maximum value in the input logLuminance channel
       store in min_logLum and max_logLum
    2) subtract them to find the range
    3) generate a histogram of all the values in the logLuminance channel using
       the formula: bin = (lum[i] - lumMin) / lumRange * numBins
    4) Perform an exclusive scan (prefix sum) on the histogram to get
       the cumulative distribution of luminance values (this should go in the
       incoming d_cdf pointer which already has been allocated for you)       */
    int threadLimit = 512;
    int blockWidth = trunc(sqrt(threadLimit));
    int blockHeight = trunc(sqrt(threadLimit));
    int numBlockRows = numRows / blockWidth + (numRows % blockWidth != 0);
    int numBlockCols = numCols / blockHeight + (numCols % blockHeight != 0);
    const dim3 blockSize(blockWidth, blockHeight, 1);
    const dim3 gridSize(numBlockRows, numBlockCols, 1);
    float *h_minValue;
    float *h_maxValue;
    float *d_maxValue;
    float *d_minValue;
    checkCudaErrors(cudaMalloc(&d_minValue,   sizeof(float)));
    checkCudaErrors(cudaMalloc(&d_maxValue,   sizeof(float)));
    find_optimum<<<gridSize, blockSize>>>(d_input,  d_minValue, true);
    find_optimum<<<gridSize, blockSize>>>(d_input,  d_maxValue, false);
    checkCudaErrors(cudaMemcpy(d_minValue, h_minValue, sizeof(float), cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(d_maxValue, h_maxValue, sizeof(float), cudaMemcpyDeviceToHost));
    float difference = h_maxValue - h_minValue;
    cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
}
//This kernel takes in three color channels and recombines them
//into one image.  The alpha channel is set to 255 to represent
//that this image has no transparency.
__global__
void findOptimum(const float* const d_input, float * optimalValue, boolean isMinimum)
{
  int rowIndex = threadIdx.x + blockDim.x * blockIdx.x;
  int colIndex = threadIdx.y + blockDim.y * blockIdx.y;
  int index = (rowIndex * numCols) + colIndex;
  if (rowIndex < numRows && colIndex < numCols) {
    unsigned char red   = redChannel[index];
    unsigned char green = greenChannel[index];
    unsigned char blue  = blueChannel[index];
    //Alpha should be 255 for no transparency
    uchar4 outputPixel = make_uchar4(red, green, blue, 255);
    outputImageRGBA[index] = outputPixel;
  }
}

void cleanup() {
  checkCudaErrors(cudaFree(d_red));
  checkCudaErrors(cudaFree(d_green));
  checkCudaErrors(cudaFree(d_blue));
  checkCudaErrors(cudaFree(d_filter));
}
