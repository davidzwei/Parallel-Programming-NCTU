__global__ void convolution(int filterWidth, float *filter, int imageHeight, int imageWidth, float *inputImage, float *outputImage)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    
    // Iterate over the rows of the source image
    int halffilterSize = filterWidth >> 1;
    float sum = 0.0f;
    int k, l;
    
    // Apply the filter to the neighborhood
    for (k = -halffilterSize; k <= halffilterSize; k++)
    {
        for (l = -halffilterSize; l <= halffilterSize; l++)
        {
            if (j + k >= 0 && j + k < imageHeight &&
                i + l >= 0 && i + l < imageWidth)
            {
                sum += inputImage[(j + k) * imageWidth + i + l] *
                       filter[(k + halffilterSize) * filterWidth + l + halffilterSize];
            }
        }
    }
    outputImage[j * imageWidth + i] = sum;
}


extern "C" void hostFE(int filterWidth, float *filter, int imageHeight, int imageWidth,
                       float *inputImage, float *outputImage)
{
    float *d_filter, *d_inputImage, *d_outputImage;
    int filterSize = filterWidth * filterWidth * sizeof(float);
    int inputImageSize = imageHeight * imageWidth * sizeof(int);
    int outputImageSize = inputImageSize;

    cudaMalloc(&d_filter, filterSize);
    cudaMalloc(&d_inputImage, inputImageSize);
    cudaMalloc(&d_outputImage, outputImageSize);

    // cp mem to device
    cudaMemcpy(d_filter, filter, filterSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_inputImage, inputImage, inputImageSize, cudaMemcpyHostToDevice);

    int block_size = 16;

    dim3 threadsPerBlock(block_size, block_size);
    dim3 numBlocks(imageWidth / block_size, imageHeight / block_size);
    // kernel
    convolution<<<threadsPerBlock, numBlocks>>>(filterWidth, d_filter, imageHeight, imageWidth, d_inputImage, d_outputImage);
    
    // cp mem to host
    cudaMemcpy(outputImage, d_outputImage, outputImageSize, cudaMemcpyDeviceToHost);
    
    // free mem
    cudaFree(d_outputImage);
    cudaFree(d_inputImage);
    cudaFree(d_filter);
}
