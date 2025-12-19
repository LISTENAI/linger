#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>

constexpr float NEG_LN2 = -0.69314718055994530941723212145818;

template <typename scalar_t>
__global__ void fake_quant_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    scalar_t* __restrict__ mask,
    const float scale,
    const float quant_min,
    const float quant_max,
    const int64_t numel) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numel) {
        float x_s = static_cast<float>(input[idx]) * scale;
        // 量化 (四舍五入)
        float q = floorf(x_s + 0.5f);
        // clamp
        q = fminf(fmaxf(q, (float)quant_min), (float)quant_max);
        // clamp的mask,用于backward
        bool is_clamped = (q == (float)quant_min) || (q == (float)quant_max);
        mask[idx] = static_cast<scalar_t>(is_clamped);
        // 反量化
        output[idx] = static_cast<scalar_t>(q / scale);
        // if ( (idx % 10000) ==0){
        //     printf("[%d], x_s=%f, q = %f, output=%f \r\n", idx, x_s, q, output[idx]);
        // }
    }
}


std::tuple<torch::Tensor, torch::Tensor, float> fake_quant_cuda(
    torch::Tensor input,
    int bit,
    float factor,
    float scale_min,
    float quant_min,
    float quant_max) {
    // printf("到位置 2 了 \n");
    // 计算 scale
    float f = (float)(bit - 1) - factor;
    float scale = powf(2.0f, roundf(f));
    // scale = fminf(fmaxf(scale, 1e-6f), powf(2.0f, 32));
    // printf("到位置 1 了,scale:%f \n", scale);
    auto output = torch::empty_like(input);
    auto mask = torch::empty_like(input); 
    //, input.options().dtype(torch::kBool)); //kBool就是uint8_t
    const int threads = 256;
    const int blocks = (input.numel() + threads - 1) / threads;

    if (scale > scale_min){
        scale = scale_min;
    }
    // printf("到位置 2 了,scale:%f \n", scale);
    // auto output_cpu = output.to(torch::kCPU);
    // printf("scale:%f \n", scale);
    // printf("output开始: \n");
    // std::cout << output_cpu << std::endl;
    // printf("output结束: \n");
    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "fake_quant_cuda", ([&] {
        fake_quant_kernel<scalar_t><<<blocks, threads>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            mask.data_ptr<scalar_t>(),
            scale,
            quant_min,
            quant_max,
            input.numel());
    }));
    // cudaDeviceSynchronize();
    // auto output_cpu2 = output.to(torch::kCPU);
    // printf("output开始: \n");
    // std::cout << output_cpu2 << std::endl;
    // printf("output结束: \n");
    return std::make_tuple(output, mask, scale);
}


std::tuple<torch::Tensor, torch::Tensor> bias_quant_cuda(
    torch::Tensor input,
    int bit,
    float scale,
    float scale_min,
    float quant_min,
    float quant_max) {
    
    auto output = torch::empty_like(input);
    auto mask = torch::empty_like(input); 
    //, input.options().dtype(torch::kBool)); //kBool就是uint8_t
    const int threads = 256;
    const int blocks = (input.numel() + threads - 1) / threads;

    if (scale > scale_min){
        scale = scale_min;
    }

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "fake_quant_cuda", ([&] {
        fake_quant_kernel<scalar_t><<<blocks, threads>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            mask.data_ptr<scalar_t>(),
            scale,
            quant_min,
            quant_max,
            input.numel());
    }));

    return std::make_tuple(output, mask);
}

// q_x = (x * s).round().clamp() / s
// 记r=(x * s).round().clamp(), 则q_x对s的导数为:
// =( r对s求导/s ) - r / s^2
// (1) clamp操作未触发
// = x / s - r / s^2
// = x / s - q_x / s
// = (x - q_x) / s
// (2)clamp操作触发
// = - q_x / s

// 从s反向传播到learning_data需乘上 -ln2 * scale

template <typename scalar_t>
__global__ void fake_quant_kernel_with_grad_scale(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    scalar_t* __restrict__ mask,
    scalar_t* __restrict__ scale_coff_back,
    const float scale,
    const float learning_data_coff,
    const float quant_min,
    const float quant_max,
    const int64_t numel) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numel) {
        float x_s = static_cast<float>(input[idx]) * scale;
        // 量化 (四舍五入)
        float q = floorf(x_s + 0.5f);
        // clamp
        q = fminf(fmaxf(q, (float)quant_min), (float)quant_max);
        // clamp的mask,用于backward
        bool is_clamped = ((q == quant_min) || (q == quant_max));
        
        mask[idx] = static_cast<scalar_t>(is_clamped);
        // 反量化
        output[idx] = static_cast<scalar_t>(q / scale);
        // if ( (idx % 10000) ==0){
        //     printf("[%d], x_s=%f, q = %f, output=%f \r\n", idx, x_s, q, output[idx]);
        // }

        scale_coff_back[idx] = static_cast<scalar_t>((mask[idx] * (-output[idx] * learning_data_coff / scale ) + (1-mask[idx]) * (input[idx] - output[idx]) * learning_data_coff / scale));
        // if (is_clamped){
        //     scale_coff_back[idx] = static_cast<scalar_t>(-output[idx] * learning_data_coff / scale );
        // }else{
        //     scale_coff_back[idx] = static_cast<scalar_t>(input[idx] - output[idx]) * learning_data_coff / scale;
        // }
    }
}



std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, float> fake_quant_cuda_with_grad_scale(
    torch::Tensor input,
    int bit,
    float factor,
    float scale_min,
    float quant_min,
    float quant_max) {
    // printf("到位置 2 了 \n");
    // 计算 scale
    float f = (float)(bit - 1) - factor;
    float scale = powf(2.0f, roundf(f));
    // scale = fminf(fmaxf(scale, 1e-6f), powf(2.0f, 32));
    // printf("到位置 1 了,scale:%f \n", scale);
    auto output = torch::empty_like(input);
    auto mask = torch::empty_like(input); 
    auto scale_coff_back = torch::empty_like(input);
    //, input.options().dtype(torch::kBool)); //kBool就是uint8_t
    const int threads = 256;
    const int blocks = (input.numel() + threads - 1) / threads;

    if (scale > scale_min){
        scale = scale_min;
    }
    // printf("到位置 2 了,scale:%f \n", scale);
    // auto output_cpu = output.to(torch::kCPU);
    // printf("scale:%f \n", scale);
    // printf("output开始: \n");
    // std::cout << output_cpu << std::endl;
    // printf("output结束: \n");
    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "fake_quant_cuda", ([&] {
        fake_quant_kernel_with_grad_scale<scalar_t><<<blocks, threads>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            mask.data_ptr<scalar_t>(),
            scale_coff_back.data_ptr<scalar_t>(),
            scale,
            NEG_LN2 * scale,
            quant_min,
            quant_max,
            input.numel());
    }));
    // cudaDeviceSynchronize();
    // auto output_cpu2 = output.to(torch::kCPU);
    // printf("output开始: \n");
    // std::cout << output_cpu2 << std::endl;
    // printf("output结束: \n");
    return std::make_tuple(output, mask, scale_coff_back, scale);
}


std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> bias_quant_cuda_with_grad_scale(
    torch::Tensor input,
    int bit,
    float scale,
    float scale_min,
    float quant_min,
    float quant_max) {
    
    auto output = torch::empty_like(input);
    auto mask = torch::empty_like(input); 
    auto scale_coff_back = torch::empty_like(input);
    //, input.options().dtype(torch::kBool)); //kBool就是uint8_t
    const int threads = 256;
    const int blocks = (input.numel() + threads - 1) / threads;

    if (scale > scale_min){
        scale = scale_min;
    }

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "fake_quant_cuda", ([&] {
        fake_quant_kernel_with_grad_scale<scalar_t><<<blocks, threads>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            mask.data_ptr<scalar_t>(),
            scale_coff_back.data_ptr<scalar_t>(),
            scale,
            1,
            quant_min,
            quant_max,
            input.numel());
    }));

    return std::make_tuple(output, mask, scale_coff_back);
}


