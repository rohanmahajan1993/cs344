// Homework 2
// Image Blurring
//
// In this homework we are blurring an image. To do this, imagine that we have
// a square array of weight values. For each pixel in the image, imagine that we
// overlay this square array of weights on top of the image such that the center
// of the weight array is aligned with the current pixel. To compute a blurred
// pixel value, we multiply each pair of numbers that line up. In other words, we
// multiply each weight with the pixel underneath it. Finally, we add up all of the
// multiplied numbers and assign that value to our output for the current pixel.
// We repeat this process for all the pixels in the image.

// To help get you started, we have included some useful notes here.

//****************************************************************************

// For a color image that has multiple channels, we suggest separating
// the different color channels so that each color is stored contiguously
// instead of being interleaved. This will simplify your code.

// That is instead of RGBARGBARGBARGBA... we suggest transforming to three
// arrays (as in the previous homework we ignore the alpha channel again):
//  1) RRRRRRRR...
//  2) GGGGGGGG...
//  3) BBBBBBBB...
//
// The original layout is known an Array of Structures (AoS) whereas the
// format we are converting to is known as a Structure of Arrays (SoA).

// As a warm-up, we will ask you to write the kernel that performs this
// separation. You should then write the "meat" of the assignment,
// which is the kernel that performs the actual blur. We provide code that
// re-combines your blurred results for each color channel.

//****************************************************************************

// You must fill in the gaussian_blur kernel to perform the blurring of the
// inputChannel, using the array of weights, and put the result in the outputChannel.

// Here is an example of computing a blur, using a weighted average, for a single
// pixel in a small image.
//
// Array of weights:
//
//  0.0  0.2  0.0
//  0.2  0.2  0.2
//  0.0  0.2  0.0
//
// Image (note that we align the array of weights to the center of the box):
//
//    1  2  5  2  0  3
//       -------
//    3 |2  5  1| 6  0       0.0*2 + 0.2*5 + 0.0*1 +
//      |       |
//    4 |3  6  2| 1  4   ->  0.2*3 + 0.2*6 + 0.2*2 +   ->  3.2
//      |       |
//    0 |4  0  3| 4  2       0.0*4 + 0.2*0 + 0.0*3
//       -------
//    9  6  5  0  3  9
//
//         (1)                         (2)                 (3)
//
// A good starting place is to map each thread to a pixel as you have before.
// Then every thread can perform steps 2 and 3 in the diagram above
// completely independently of one another.

// Note that the array of weights is square, so its height is the same as its width.
// We refer to the array of weights as a filter, and we refer to its width with the
// variable filterWidth.

//****************************************************************************

// Your homework submission will be evaluated based on correctness and speed.
// We test each pixel against a reference solution. If any pixel differs by
// more than some small threshold value, the system will tell you that your
// solution is incorrect, and it will let you try again.

// Once you have gotten that working correctly, then you can think about using
// shared memory and having the threads cooperate to achieve better performance.

//****************************************************************************

// Also note that we've supplied a helpful debugging function called checkCudaErrors.
// You should wrap your allocation and copying statements like we've done in the
// code we're supplying you. Here is an example of the unsafe way to allocate
// memory on the GPU:
//
// cudaMalloc(&d_red, sizeof(unsigned char) * numRows * numCols);
//
// Here is an example of the safe way to do the same thing:
//
// checkCudaErrors(cudaMalloc(&d_red, sizeof(unsigned char) * numRows * numCols));
//
// Writing code the safe way requires slightly more typing, but is very helpful for
// catching mistakes. If you write code the unsafe way and you make a mistake, then
// any subsequent kernels won't compute anything, and it will be hard to figure out
// why. Writing code the safe way will inform you as soon as you make a mistake.

// Finally, remember to free the memory you allocate at the end of the function.

//****************************************************************************

#include "reference_calc.cpp"
#include "utils.h"

__global__
void gaussian_blur(const unsigned char* const inputChannel,
                   unsigned char* const outputChannel,
                   int numRows, int numCols,
                   const float* const filter, const int filterWidth)
{
  int rowIndex = threadIdx.x + blockDim.x * blockIdx.x;
  int colIndex = threadIdx.y + blockDim.y * blockIdx.y;
  int index = (rowIndex * numCols) + colIndex;
  if (rowIndex < numRows && colIndex < numCols) {
   float result = 0.f;
      //For every value in the filter around the pixel (c, r)
      for (int filter_r = -filterWidth/2; filter_r <= filterWidth/2; ++filter_r) {
        for (int filter_c = -filterWidth/2; filter_c <= filterWidth/2; ++filter_c) {
          //Find the global image position for this filter position
          //clamp to boundary of the image
          int image_r = min(max(rowIndex + filter_r, 0), static_cast<int>(numRows - 1));
          int image_c = min(max(colIndex + filter_c, 0), static_cast<int>(numCols - 1));
          float image_value = static_cast<float>(inputChannel[image_r * numCols + image_c]);
          float filter_value = filter[(filter_r + filterWidth/2) * filterWidth + filter_c + filterWidth/2];
          result += image_value * filter_value;
        }
      }
      outputChannel[index] = result;
  }
}

//This kernel takes in an image represented as a uchar4 and splits
//it into three images consisting of only one color channel each
__global__
void separateChannels(const uchar4* const inputImageRGBA,
                      int numRows,
                      int numCols,
                      unsigned char* const redChannel,
                      unsigned char* const greenChannel,
                      unsigned char* const blueChannel)
{
  int rowIndex = threadIdx.x + blockDim.x * blockIdx.x;
  int colIndex = threadIdx.y + blockDim.y * blockIdx.y;
  int index = (rowIndex * numCols) + colIndex;
  if (rowIndex < numRows && colIndex < numCols) {
    uchar4 rgba = inputImageRGBA[index];
    redChannel[index] = rgba.x;
    greenChannel[index] = rgba.y;
    blueChannel[index] = rgba.z;
  }
}

//This kernel takes in three color channels and recombines them
//into one image.  The alpha channel is set to 255 to represent
//that this image has no transparency.
__global__
void recombineChannels(const unsigned char* const redChannel,
                       const unsigned char* const greenChannel,
                       const unsigned char* const blueChannel,
                       uchar4* const outputImageRGBA,
                       int numRows,
                       int numCols)
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

unsigned char *d_red, *d_green, *d_blue;
float         *d_filter;

void allocateMemoryAndCopyToGPU(const size_t numRowsImage, const size_t numColsImage,
                                const float* const h_filter, const size_t filterWidth)
{
  checkCudaErrors(cudaMalloc(&d_red,   sizeof(unsigned char) * numRowsImage * numColsImage));
  checkCudaErrors(cudaMalloc(&d_green, sizeof(unsigned char) * numRowsImage * numColsImage));
  checkCudaErrors(cudaMalloc(&d_blue,  sizeof(unsigned char) * numRowsImage * numColsImage));

  int numFilterBytes = sizeof(float) * filterWidth * filterWidth;
  checkCudaErrors(cudaMalloc(&d_filter, numFilterBytes));
  checkCudaErrors(cudaMemcpy(d_filter, h_filter, numFilterBytes, cudaMemcpyHostToDevice));
}

void your_gaussian_blur(const uchar4 * const h_inputImageRGBA, uchar4 * const d_inputImageRGBA,
                        uchar4* const d_outputImageRGBA, const size_t numRows, const size_t numCols,
                        unsigned char *d_redBlurred, 
                        unsigned char *d_greenBlurred, 
                        unsigned char *d_blueBlurred,
                        const int filterWidth)
{
  int threadLimit = 512;
  int blockWidth = trunc(sqrt(threadLimit));
  int blockHeight = trunc(sqrt(threadLimit));
  int numBlockRows = numRows / blockWidth + (numRows % blockWidth != 0);
  int numBlockCols = numCols / blockHeight + (numCols % blockHeight != 0);
  const dim3 blockSize(blockWidth, blockHeight, 1);
  const dim3 gridSize(numBlockRows, numBlockCols, 1);
  separateChannels<<<gridSize, blockSize>>>(d_inputImageRGBA, numRows, numCols, d_red, d_green, d_blue);

  cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
  gaussian_blur<<<gridSize, blockSize>>>(d_red, d_redBlurred,numRows, numCols, d_filter, filterWidth);
  gaussian_blur<<<gridSize, blockSize>>>(d_green, d_greenBlurred,numRows, numCols, d_filter, filterWidth);
  gaussian_blur<<<gridSize, blockSize>>>(d_blue, d_blueBlurred,numRows, numCols, d_filter, filterWidth);
  cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
 
  recombineChannels<<<gridSize, blockSize>>>(d_redBlurred,
                                             d_greenBlurred,
                                             d_blueBlurred,
                                             d_outputImageRGBA,
                                             numRows,
                                             numCols);
  cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
}

void cleanup() {
  checkCudaErrors(cudaFree(d_red));
  checkCudaErrors(cudaFree(d_green));
  checkCudaErrors(cudaFree(d_blue));
  checkCudaErrors(cudaFree(d_filter));
}
