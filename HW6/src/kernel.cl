__kernel void convolution(
    int filterWidth, __constant float *filter, int imageHeight, int imageWidth, __global const char *inputImage, __global float *outputImage) 
 {
    int halfFilterSize = filterWidth / 2;
    float sum = 0.0;

    // image x, y
    int x = get_global_id(0); 
    int y = get_global_id(1);

    // filter x,y start, end
    int row_start = y - halfFilterSize >= 0 ? 0 : halfFilterSize - y;
    int row_end = y + halfFilterSize < imageHeight ? filterWidth - 1 : imageHeight - y;
    int col_start = x - halfFilterSize >= 0 ? 0 : halfFilterSize - x;
    int col_end = x + halfFilterSize < imageWidth ? filterWidth - 1 : imageWidth - x;

    // filter based
    for (int i = row_start; i <= row_end; i++) {
        int row = y - halfFilterSize + i;
        int col = x - halfFilterSize;
        for (int j = col_start; j <= col_end; j++) {
            sum += inputImage[row * imageWidth + col + j] * filter[i * filterWidth + j];
        }
    }
    outputImage[y * imageWidth + x] = sum;
}
