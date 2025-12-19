#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <stdint.h>
#include <math.h>
#include <cmath>

#define MAX_BITS(bits) ((1LL << (bits - 1)) - 1)
#define MIN_BITS(bits) (-(1LL << (bits - 1)))
#define SATURATE(x, bits)                \
  ((x) > MAX_BITS(bits) ? MAX_BITS(bits) \
                        : ((x) < MIN_BITS(bits) ? MIN_BITS(bits) : (x)))

static __device__ int32_t shift_pure(int64_t v, int32_t s)
{
	if (s >= 63 || s <= -63) {
		return 0;
	}
	if (s > 0) {
		v = v << s;
	} else {
		v = v >> (-s);
	}
	return SATURATE(v, 32);
}

static __device__ int32_t sub32s(int32_t a, int32_t b)
{
	int64_t s = 0;
	s = (int64_t)((int64_t)a - (int64_t)b);
	return SATURATE(s, 32);
}

static __device__ int32_t add32s(int32_t a, int32_t b)
{
	int64_t s = 0;
	s = (int64_t)((int64_t)a + (int64_t)b);
	return SATURATE(s, 32);
}

static __device__ int32_t shift_rasym(int64_t x,int32_t n)
{
	// double y = (double)x * (double)pow(2, n);
	// y = floor(y + 0.5);
	// return (int32_t)y;
	if (n >= 63 || n <= -63) {
		return 0;
	}
    if (n >= 0) {
		x = x << n;
	} 
	else {
        n = (-n);
        x = x >> (n - 1);
		x = (x & 0x1) + (x >> 1);
    }
    return x;
}

static __device__ int32_t shift_rasyms(int64_t x, int32_t n)
{
	// double y = (double)x * (double)pow(2, n);
	
	// y = floor(y + 0.5);
	// return SATURATE(y, 32);
	if (n >= 64 || n <= -64) {
		return 0;
	}
    if (n >= 0) {
		x = x << n;
	} 
	else {
        n = (-n);
        x = x >> (n - 1);
		x = (x & 0x1) + (x >> 1);
    }
	return SATURATE(x, 32);
}

static __device__ int64_t shfit_floor_x05_int64(int64_t x, int32_t shift)
{
	int64_t val = x;

	if (shift >= 64) {
		return 0;
	}
	if (shift > 0) {
		val = val >> (shift - 1);
		val = (val & 0x1) + (val >> 1);
	}

	return val;
}

static __device__ void arcs_qsoftmax_c(const int* __restrict__ x_ptr, int* __restrict__ y_ptr, int32_t N, int32_t* p23)
{
	int32_t max_value = x_ptr[0];
	int32_t X = 0;
	int64_t Y = 0;
	int32_t E = 0;
	int64_t E_SUM = 0;
	uint32_t A = 0x800000, B = 0, C = 0;
	for (int i = 1; i < N; i++)
	{
		max_value = x_ptr[i] > max_value ? x_ptr[i] : max_value;
	}

	for (int i = 0; i < N; i++)
	{
		X = sub32s(x_ptr[i] ,max_value);
		X = shift_rasyms((int64_t)X * (int64_t)774541002,-31);
		E = X >> 23;
		E = E + 1;

		X = X & 0x7fffff;
		X = X - 0x800000;

		Y = p23[0];

		for (int j = 1; j < 5; j++)
		{
			Y = shift_rasym((int64_t)Y * (int64_t)X,-23) + p23[j];
		}
		
		// y_ptr[i] = (int32_t)(Y  * pow(2, E));
		y_ptr[i] = shift_pure(Y, E);
	}

	for (int i = 0; i < N; i++)
	{
		E_SUM += y_ptr[i];
	}

	B = SATURATE(E_SUM, 32);
	for (int i = 1; i <= 30; i++) {
		if(A>=B) {
			C = C+1; 
			C = C*2;
			A = A-B;
			A = A*2;
		} else {
			C = C*2;
			A = A*2;
		}
	}

	for (int i = 0; i < N; i++)
	{
		y_ptr[i] = shfit_floor_x05_int64((int64_t)C * (int64_t)y_ptr[i], 38);
	}
}

__global__ void arcs_qsoftmax_gpu_kernel(const int* __restrict__ p_in,
	int* __restrict__ p_out, 
	int32_t N, int32_t L, int32_t* p23)
{
	int32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < L)
	// for (int k = 0; k < L; k++)
	{
		const int* x_ptr = p_in + N * idx;
		int * y_ptr = p_out + N * idx;

		arcs_qsoftmax_c(x_ptr, y_ptr, N, p23);
	}	
}

torch::Tensor arcs_qsoftmax_gpu(const torch::Tensor& in, int64_t dim)
{
	// int32_t N = in.numel();
	int32_t N = in.size(1);
	int32_t L = in.size(0);
	const dim3 threads = (64);
    const dim3 blocks((L + threads.x - 1) / threads.x);
	auto out = torch::zeros_like(in);
	const int* x_ptr = in.data_ptr<int>();
	int * y_ptr = out.data_ptr<int>();

	// static const int32_t p[] = { 14685058, 114217091, 514075394, 1488269031, 2147475316 };
	static const int32_t p23[] = { 57364 ,446161 ,2008107 , 5813551, 8388575 };

	// int32_t *buffer_p;
	int32_t *buffer_p23;

	// cudaMalloc((void **)&buffer_p, 5 * sizeof(int32_t));
	// cudaMemcpy(buffer_p, p, 5 * sizeof(int32_t), cudaMemcpyHostToDevice);
	cudaMalloc((void **)&buffer_p23, 5 * sizeof(int32_t));
	cudaMemcpy(buffer_p23, p23, 5 * sizeof(int32_t), cudaMemcpyHostToDevice);

	arcs_qsoftmax_gpu_kernel<<<blocks, threads>>>(x_ptr, y_ptr, N, L, buffer_p23);
	// arcs_qsoftmax_gpu_kernel<<<1, 1>>>(x_ptr, y_ptr, N, L, buffer_p, buffer_p23);
	
	// cudaFree(buffer_p);
	cudaFree(buffer_p23);
	
	return out;
}

