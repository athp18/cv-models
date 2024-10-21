#include <torch/extension.h>

__global__ void conv2d_kernel(float* input, float* output, float* kernel, 
                              int input_channels, int output_channels, 
                              int input_height, int input_width, 
                              int kernel_size, int stride, int padding) {
    int batch = blockIdx.x;
    int out_y = blockIdx.y * blockDim.y + threadIdx.y;
    int out_x = blockIdx.z * blockDim.x + threadIdx.x;

    if (out_y >= input_height || out_x >= input_width) return;

    // Perform 2D convolution for each output channel
    for (int out_ch = 0; out_ch < output_channels; ++out_ch) {
        float output_value = 0.0;
        for (int in_ch = 0; in_ch < input_channels; ++in_ch) {
            for (int ky = 0; ky < kernel_size; ++ky) {
                for (int kx = 0; kx < kernel_size; ++kx) {
                    int in_y = out_y * stride + ky - padding;
                    int in_x = out_x * stride + kx - padding;

                    if (in_y >= 0 && in_x >= 0 && in_y < input_height && in_x < input_width) {
                        int input_idx = ((batch * input_channels + in_ch) * input_height + in_y) * input_width + in_x;
                        int kernel_idx = ((out_ch * input_channels + in_ch) * kernel_size + ky) * kernel_size + kx;
                        output_value += input[input_idx] * kernel[kernel_idx];
                    }
                }
            }
        }
        int output_idx = ((batch * output_channels + out_ch) * input_height + out_y) * input_width + out_x;
        output[output_idx] = output_value;
    }
}
