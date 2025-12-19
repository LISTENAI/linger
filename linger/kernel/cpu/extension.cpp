#include <torch/extension.h>
#include <torch/types.h>
#include <vector>
#include <iostream>
#include <tuple>

void find_table_cpu(torch::Tensor value, torch::Tensor table, torch::Tensor table_index);
void find_table_gpu(torch::Tensor value, torch::Tensor table, torch::Tensor table_index);
void find_table(torch::Tensor value, torch::Tensor table, torch::Tensor table_index)
{
    if (value.device().type() == torch::kCUDA)
    {
        return find_table_gpu(value, table, table_index);
    }
    else
    {
        return find_table_cpu(value, table, table_index);
    }
}

torch::Tensor venusa_qsigmoid_cpu(torch::Tensor a);
torch::Tensor venusa_qsigmoid_gpu(torch::Tensor a);
torch::Tensor venusa_qsigmoid_forward(torch::Tensor a)
{
    if (a.device().type() == torch::kCUDA)
    {
        return venusa_qsigmoid_gpu(a);
    }
    else
    {
        return venusa_qsigmoid_cpu(a);
    }
}

torch::Tensor arcs_qsigmoid_cpu(torch::Tensor a);
torch::Tensor arcs_qsigmoid_gpu(torch::Tensor a);
torch::Tensor arcs_qsigmoid_forward(torch::Tensor a)
{
    if (a.device().type() == torch::kCUDA)
    {
        return arcs_qsigmoid_gpu(a);
    }
    else
    {
        return arcs_qsigmoid_cpu(a);
    }
}

torch::Tensor venusa_qtanh_cpu(torch::Tensor a);
torch::Tensor venusa_qtanh_gpu(torch::Tensor a);
torch::Tensor venusa_qtanh_forward(torch::Tensor a)
{
    if (a.device().type() == torch::kCUDA)
    {
        return venusa_qtanh_gpu(a);
    }
    else
    {
        return venusa_qtanh_cpu(a);
    }
}

torch::Tensor arcs_qtanh_cpu(torch::Tensor a);
torch::Tensor arcs_qtanh_gpu(torch::Tensor a);
torch::Tensor arcs_qtanh_forward(torch::Tensor a)
{
    if (a.device().type() == torch::kCUDA)
    {
        return arcs_qtanh_gpu(a);
    }
    else
    {
        return arcs_qtanh_cpu(a);
    }
}

torch::Tensor arcs_qsoftmax_cpu(const torch::Tensor& in, int64_t dim);
torch::Tensor arcs_qsoftmax_gpu(const torch::Tensor& in, int64_t dim);
torch::Tensor arcs_qsoftmax_forward(const torch::Tensor& in, int64_t dim)
{
    if (in.device().type() == torch::kCUDA)
    {
        return arcs_qsoftmax_gpu(in, dim);
    }
    else
    {
        return arcs_qsoftmax_cpu(in, dim);
    }
}

torch::Tensor venusa_qsoftmax_cpu(const torch::Tensor& in, int64_t dim);
torch::Tensor venusa_qsoftmax_gpu(const torch::Tensor& in, int64_t dim);
torch::Tensor venusa_qsoftmax_forward(const torch::Tensor& in, int64_t dim)
{
    if (in.device().type() == torch::kCUDA)
    {
        return venusa_qsoftmax_gpu(in, dim);
    }
    else
    {
        return venusa_qsoftmax_cpu(in, dim);
    }
}

torch::Tensor qlayernorm_kernel_cpu(torch::Tensor numerator, torch::Tensor denominator, float scale_x);
torch::Tensor qlayernorm_kernel_gpu(torch::Tensor numerator, torch::Tensor denominator, float scale_x);
torch::Tensor qlayernorm_kernel_forward(torch::Tensor numerator, torch::Tensor denominator, float scale_x)
{
    if (numerator.device().type() == torch::kCUDA)
    {
        return qlayernorm_kernel_gpu(numerator, denominator, scale_x);
    }
    else
    {
        return qlayernorm_kernel_cpu(numerator, denominator, scale_x);
    }
}



std::tuple<torch::Tensor, torch::Tensor, float> fake_quant_cuda(torch::Tensor input,int bit,float factor,float scale_min, float quant_min,float quant_max);

std::tuple<torch::Tensor, torch::Tensor, float> fake_quant(
    torch::Tensor input,
    int bit,
    float factor,
    float scale_min,
    float quant_min,
    float quant_max) {
    // return fake_quant_cuda(input, bit, factor, quant_min, quant_max);
    // printf("到位置 1 了 \n");
    if (input.device().type() == torch::kCUDA){
        return fake_quant_cuda(input, bit, factor, scale_min, quant_min, quant_max);
    }
    else{
        throw std::runtime_error("尚未实现cpu版本伪量化，请使用python版本——NATIVE模式");
    }
}

std::tuple<torch::Tensor, torch::Tensor> bias_quant_cuda(torch::Tensor input,int bit,float scale,float scale_min, float quant_min,float quant_max);


std::tuple<torch::Tensor, torch::Tensor> bias_quant(
    torch::Tensor input,
    int bit,
    float scale,
    float scale_min,
    float quant_min,
    float quant_max) {
    // return bias_quant_cuda(input, bit, scale, quant_min, quant_max);
    // printf("到位置 1 了 \n");
    if (input.device().type() == torch::kCUDA){
        return bias_quant_cuda(input, bit, scale, scale_min, quant_min, quant_max);
    }
    else{
        throw std::runtime_error("尚未实现cpu版本伪量化，请使用python版本——NATIVE模式");
    }
}



std::tuple<torch::Tensor, torch::Tensor, torch::Tensor,float> fake_quant_cuda_with_grad_scale(torch::Tensor input,int bit,float factor,float scale_min, float quant_min,float quant_max);
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor,float> fake_quant_with_grad_scale(
    torch::Tensor input,
    int bit,
    float factor,
    float scale_min,
    float quant_min,
    float quant_max) {
    // return fake_quant_cuda(input, bit, factor, quant_min, quant_max);
    // printf("到位置 1 了 \n");
    if (input.device().type() == torch::kCUDA){
        return fake_quant_cuda_with_grad_scale(input, bit, factor, scale_min, quant_min, quant_max);
    }
    else{
        throw std::runtime_error("尚未实现cpu版本伪量化，请使用python版本——NATIVE模式");
    }
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> bias_quant_cuda_with_grad_scale(torch::Tensor input,int bit,float scale,float scale_min, float quant_min,float quant_max);
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> bias_quant_with_grad_scale(
    torch::Tensor input,
    int bit,
    float scale,
    float scale_min,
    float quant_min,
    float quant_max) {
    // return bias_quant_cuda(input, bit, scale, quant_min, quant_max);
    // printf("到位置 1 了 \n");
    if (input.device().type() == torch::kCUDA){
        return bias_quant_cuda_with_grad_scale(input, bit, scale, scale_min, quant_min, quant_max);
    }
    else{
        throw std::runtime_error("尚未实现cpu版本伪量化，请使用python版本——NATIVE模式");
    }
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("find_table", &find_table, "find_table(CPU/GPU)");
    m.def("arcs_qsoftmax_forward", &arcs_qsoftmax_forward, "arcs_qsoftmax_forward(CPU/GPU)");
    m.def("venusa_qsoftmax_forward", &venusa_qsoftmax_forward, "venusa_qsoftmax_forward(CPU/GPU)");
    m.def("venusa_qsigmoid_forward", &venusa_qsigmoid_forward, "venusa_qsigmoid_forward(CPU/GPU)");
    m.def("arcs_qsigmoid_forward", &arcs_qsigmoid_forward, "arcs_qsigmoid_forward(CPU/GPU)");
    m.def("venusa_qtanh_forward", &venusa_qtanh_forward, "venusa_qtanh_forward(CPU/GPU)");
    m.def("arcs_qtanh_forward", &arcs_qtanh_forward, "arcs_qtanh_forward(CPU/GPU)");
    m.def("qlayernorm_kernel_forward", &qlayernorm_kernel_forward, "qlayernorm_kernel_forward(CPU/GPU)");

    m.def("fake_quant", &fake_quant, "Fake Quantization (CUDA)");
    m.def("bias_quant", &bias_quant, "Bias Quantization (CUDA)");

    m.def("fake_quant_with_grad_scale", &fake_quant_with_grad_scale, "Fake Quantization With Grad Scale (CUDA)");
    m.def("bias_quant_with_grad_scale", &bias_quant_with_grad_scale, "Bias Quantization With Grad Scale (CUDA)");
}