#include "reference_calc.cpp"
#include "utils.h"

static inst const threadLimit = 512;

// Adapted from udacity code snippets
__global__ void find_optimum(float * d_out, float * d_in, bool isMinimum, int numEntries)
{
    int myId = threadIdx.x + blockDim.x * blockIdx.x;
    if (myId >= numEntries) {
	return;
    }
    int tid  = threadIdx.x;

    // do reduction in global mem
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1)
    {
        if (tid < s)
        {
	    if (isMinimum) {
              d_in[myId] = min(d_in[myId + s], d_in[myId]);
            } else {
              d_in[myId] = max(d_in[myId + s], d_in[myId]);
            }
        }
        __syncthreads();        // make sure all adds at one stage are done!
    }

    // only thread 0 writes result for this block back to global mem
    if (tid == 0)
    {
        d_out[blockIdx.x] = d_in[myId];
    }
}

float calculateDifference(const float * const d_immutable_input, int numEntries, float * h_minValue) {
    float *h_maxValue;
    float *d_input;
    float *d_intermediate_out;
    float *d_out;
    int blockWidth = threadLimit;
    int numBlocks = numEntries / threadLimit + (numBlocks % threadLimit != 0); 
    checkCudaErrors(cudaMalloc(&d_input,   sizeof(float) * numEntries));
    checkCudaErrors(cudaMalloc(&d_intermediate_out,   sizeof(float) * numBlocks));
    
    checkCudaErrors(cudaMemcpy(d_input, d_immutable_input, sizeof(float) * numEntries, cudaMemcpyDeviceToDevice));
    find_optimum<<<numBlocks, blockWidth>>>(d_intermediate_out, d_input, true, numEntries);
    find_optimum<<<1, numBlocks>>>(d_out, d_intermediate_out, true, numEntries);
    checkCudaErrors(cudaMemcpy(h_minValue, d_out, sizeof(float), cudaMemcpyDeviceToHost));

    checkCudaErrors(cudaMemcpy(d_input, d_immutable_input, sizeof(float) * numEntries, cudaMemcpyHostToDevice));
    find_optimum<<<numBlocks, blockWidth>>>(d_intermediate_out, d_input, false, numEntries);
    find_optimum<<<1, numBlocks>>>(d_out, d_intermediate_out, false, numEntries);
    checkCudaErrors(cudaMemcpy(h_maxValue, d_out, sizeof(float), cudaMemcpyDeviceToHost));

    checkCudaErrors(cudaFree(d_input));
    checkCudaErrors(cudaFree(d_intermediate_out));
    checkCudaErrors(cudaFree(d_out));

    cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
    float difference = *h_maxValue - *h_minValue; 
    return difference;
}
// Adapted from udacity code snippet
__global__ void simple_histo(int *d_bins, int *d_in, float min, float range, const int numBins, numEntries)
{
    int myId = threadIdx.x + blockDim.x * blockIdx.x;
    if (myId >= numEntries) {
      return;
    }
    float value = d_in[myId];
    unsigned int bin = min(static_cast<unsigned int>(numBins - 1),
                           static_cast<unsigned int>((value - min) / range * numBins));
    atomicAdd(&(d_bins[bin]), 1);
}

void histogram(const float* const d_immutable_input, float min, float range, int numBins, int numEntries, int * d_bins) {
    int blockWidth = threadLimit;
    int numBlocks = numEntries / threadLimit + (numBlocks % threadLimit != 0); 
    float *d_input;
    checkCudaErrors(cudaMalloc(&d_input,   sizeof(float) * numEntries));
    checkCudaErrors(cudaMalloc(&d_bins,   sizeof(int) * numBins));
    checkCudaErrors(cudaMemcpy(d_input, d_immutable_input, sizeof(float) * numEntries, cudaMemcpyDeviceToDevice));
    checkCudaErrors(cudaMemset(d_input, 0, sizeof(int) * numBins, cudaMemcpyDeviceToDevice));
    simple_histo<<<numBlocks, blockWidth>>>(d_bins, d_input, min, range, numBins, numEntries);
} 

void your_histogram_and_prefixsum(const float* const d_logLuminance,
                                  unsigned int* const d_cdf,
                                  float &min_logLum,
                                  float &max_logLum,
                                  const size_t numRows,
                                  const size_t numCols,
                                  const size_t numBins)
{
    int numEntries = numRows * numCols;
    float difference = calculateDifference(d_logLuminance, numEntries, minlogLum);
    int *d_bins;
    void histogram(d_logLuminance, *min_logLum, difference,  numBins, numEntries, d_bins);
    int *h_bins;
    checkCudaErrors(cudaMemcpy(h_bins, d_bins, sizeof(int) * numBins, cudaMemcpyDeviceToDevice));
    for (int i = 1; i < numBins; i++) {
	d_cdf[i] = d_cdf[i-1] + h_bins[i-1];	
    }
}
