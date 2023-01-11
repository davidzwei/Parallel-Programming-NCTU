#include <stdio.h>
#include <stdlib.h>
#include "hostFE.h"
#include "helper.h"

void hostFE(int filterWidth, float *filter, int imageHeight, int imageWidth,
            float *inputImage, float *outputImage, cl_device_id *device,
            cl_context *context, cl_program *program)
{
    int filterSize = filterWidth * filterWidth;

    // make input image from float to char
    char *newInputImage = (char *)malloc(imageHeight * imageWidth * sizeof(char));
    for (int i = 0; i < imageHeight * imageWidth; i++)
    {
        newInputImage[i] = (char)inputImage[i];
    }

    // cl declaration
    cl_int status, commandQueue_err, filter_err, output_err, input_err, kernel_err;
    cl_command_queue commandQueue;
    commandQueue = clCreateCommandQueue(*context, device[0], 0, &commandQueue_err);

    // memory
    cl_mem filterMem = clCreateBuffer(*context, 0, sizeof(float) * filterSize, NULL, &filter_err);
    cl_mem inputMem = clCreateBuffer(*context, 0, sizeof(char) * imageHeight * imageWidth, NULL, &input_err);
    cl_mem outputMem = clCreateBuffer(*context, 0, sizeof(float) * imageHeight * imageWidth, NULL, &output_err);

    // buffer
    clEnqueueWriteBuffer(commandQueue, inputMem, CL_MEM_READ_ONLY, 0, imageHeight * imageWidth * sizeof(char), newInputImage, 0, NULL, NULL);
    clEnqueueWriteBuffer(commandQueue, filterMem, CL_MEM_READ_ONLY, 0, filterSize * sizeof(float), filter, 0, NULL, NULL);

    cl_kernel kernel = clCreateKernel(*program, "convolution", &kernel_err);

    // argument
    clSetKernelArg(kernel, 0, sizeof(int), &filterWidth);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &filterMem);
    clSetKernelArg(kernel, 2, sizeof(int), &imageHeight);
    clSetKernelArg(kernel, 3, sizeof(int), &imageWidth);
    clSetKernelArg(kernel, 4, sizeof(cl_mem), &inputMem);
    clSetKernelArg(kernel, 5, sizeof(cl_mem), &outputMem);

    // nd range
    size_t globalSize[2] = {imageWidth, imageHeight};
    clEnqueueNDRangeKernel(commandQueue, kernel, 2, NULL, globalSize, NULL, 0, NULL, NULL);
    clEnqueueReadBuffer(commandQueue, outputMem, CL_TRUE, 0, sizeof(float) * imageHeight * imageWidth, outputImage, 0, NULL, NULL);

    // release
    clReleaseMemObject(outputMem);
    clReleaseMemObject(inputMem);
    // clReleaseMemObject(filterMem);
    clReleaseMemObject(commandQueue);
    clReleaseMemObject(kernel);

    free(newInputImage);
}
