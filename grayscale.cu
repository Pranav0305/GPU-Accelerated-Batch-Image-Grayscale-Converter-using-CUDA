#include <stdio.h>
#include <cuda_runtime.h>

#define CHANNELS 3  // RGB

// CUDA Kernel: Convert RGB image to Grayscale
__global__ void rgbToGrayscale(unsigned char* rgb, unsigned char* gray, int width, int height)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int totalPixels = width * height;

    if (idx < totalPixels)
    {
        int rgbIdx = idx * CHANNELS;

        unsigned char r = rgb[rgbIdx];
        unsigned char g = rgb[rgbIdx + 1];
        unsigned char b = rgb[rgbIdx + 2];

        // Grayscale formula
        gray[idx] = static_cast<unsigned char>(0.299f * r + 0.587f * g + 0.114f * b);
    }
}

int main()
{
    int width = 1024;
    int height = 1024;
    int numPixels = width * height;

    size_t rgbSize = numPixels * CHANNELS * sizeof(unsigned char);
    size_t graySize = numPixels * sizeof(unsigned char);

    // Host memory
    unsigned char* h_rgb = (unsigned char*)malloc(rgbSize);
    unsigned char* h_gray = (unsigned char*)malloc(graySize);

    // Initialize RGB image with dummy data
    for (int i = 0; i < numPixels * CHANNELS; i++)
        h_rgb[i] = rand() % 256;

    // Device memory
    unsigned char *d_rgb, *d_gray;
    cudaMalloc((void**)&d_rgb, rgbSize);
    cudaMalloc((void**)&d_gray, graySize);

    // Copy input image to device
    cudaMemcpy(d_rgb, h_rgb, rgbSize, cudaMemcpyHostToDevice);

    // Kernel launch configuration
    int threadsPerBlock = 256;
    int blocksPerGrid = (numPixels + threadsPerBlock - 1) / threadsPerBlock;

    printf("Launching kernel with %d blocks of %d threads\n", blocksPerGrid, threadsPerBlock);

    // Launch kernel
    rgbToGrayscale<<<blocksPerGrid, threadsPerBlock>>>(d_rgb, d_gray, width, height);
    cudaDeviceSynchronize();

    // Copy result back to host
    cudaMemcpy(h_gray, d_gray, graySize, cudaMemcpyDeviceToHost);

    printf("Grayscale conversion completed successfully on GPU.\n");

    // Cleanup
    cudaFree(d_rgb);
    cudaFree(d_gray);
    free(h_rgb);
    free(h_gray);

    return 0;
}
