#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <stdint.h>
#include <math.h>
#include <cmath>

#ifndef MAX_INT32
#define MAX_INT32 ((int32_t)0x7FFFFFFFL)
#endif
#ifndef MIN_INT32
#define MIN_INT32 ((int32_t)0x80000000L)
#endif


static __device__ int32_t SAT32(int64_t x)
{

	int32_t result = 0;
	if (x > MAX_INT32)
	{
		result = MAX_INT32;
	}
	else if (x < MIN_INT32)
	{
		result = MIN_INT32;
	}
	else {
		result = x;
	}
	return result;
}

static __device__ int32_t sign(double x)
{
	return x < 0 ? -1 : 1;
}

static __device__ int32_t sub32s(int32_t a, int32_t b)
{
	int64_t s = 0;
	int32_t t = 0;
	s = (int64_t)a - (int64_t)b;
	if (s > MAX_INT32)
	{
		t = MAX_INT32;
	}
	else if (s <= MIN_INT32)
	{
		t = MIN_INT32;
	}
	else
	{
		t = (int32_t)s;
	}
	return t;
}

static __device__ int32_t add32s(int32_t a, int32_t b)
{
	int64_t s = 0;
	int32_t t = 0;
	s = (int64_t)a + (int64_t)b;
	if (s > MAX_INT32)
	{
		t = MAX_INT32;
	}
	else if (s <= MIN_INT32)
	{
		t = MIN_INT32;
	}
	else
	{
		t = (int32_t)s;
	}
	return t;
}

static __device__ int32_t shift_rasym(int64_t x,int32_t n)
{
	double y = (double)x * (double)pow(2, n);
	y = floor(y + 0.5);
	return (int32_t)y;
}

static __device__ int32_t shift_rasyms(int64_t x, int32_t n)
{
	double y = (double)x * (double)pow(2, n);
	
	y = floor(y + 0.5);
	int32_t s = 0;
	if (y > MAX_INT32)
	{
		s = MAX_INT32;
	}
	else if (y <= MIN_INT32)
	{
		s = MIN_INT32;
	}
	else {
		s = (int32_t)y;
	}
	return s;
}

static __device__ int32_t nsaz(int32_t x)
{
	if (x == 0)
		return 0;
	uint32_t ux = x < 0 ? -x : x;
	if (ux == 0x80000000)
		return 0;
	ux = ux & 0x7FFFFFFF;
	int32_t ix = 0;
	while (!(ux & 0x40000000) && ix < 31)
	{
		ux = ux * 2;
		ix++;
	}
	return ix;
}

static __device__ void venus_qsoftmax_c(const int* __restrict__ x_ptr, int* __restrict__ y_ptr, int32_t N, int32_t* p, int32_t* p23)
{
	int32_t max_value = x_ptr[0];
	int32_t X = 0;
	int32_t Y = 0;
	int32_t E = 0;
	int32_t E_SUM = 0;
	
	for (int i = 0; i < N; i++)
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
		
		y_ptr[i] = (int32_t)(Y  * pow(2, E));
	}

	for (int i = 0; i < N; i++)
	{
		E_SUM = add32s(E_SUM,y_ptr[i]);
	}

	X = E_SUM;
	int32_t n = nsaz(X) - 8;
	X = n < 0 ? SAT32(((int64_t)X) >> (-n)) : SAT32(((int64_t)X) << n);
	n = n + 1;

	int32_t is_zero = X == 0 ? 1 : 0;
	Y = (int32_t)0xBB5000 - X;
	int32_t t = 0;
	for (int i = 0; i < 2; i++)
	{
		E = 0x400000 - shift_rasym((int64_t)X * (int64_t)Y, -23);
		E = E + E;
		Y = Y + shift_rasym((int64_t)Y * (int64_t)E, -23);
	}
	Y = n < 0 ? SAT32(((int64_t)Y) >> (-n)) : SAT32(((int64_t)Y) << n);
	for (int i = 0; i < N; i++)
	{
		y_ptr[i] = shift_rasyms((int64_t)Y * (int64_t)y_ptr[i], -31);
	}
}

__global__ void venus_qsoftmax_gpu_kernel(const int* __restrict__ p_in,
	int* __restrict__ p_out, 
	int32_t N, int32_t L, int32_t* p , int32_t* p23)
{
	int32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < L)
	// for (int k = 0; k < L; k++)
	{
		const int* x_ptr = p_in + N * idx;
		int * y_ptr = p_out + N * idx;

		venus_qsoftmax_c(x_ptr, y_ptr, N, p, p23);
	}	
}

torch::Tensor venus_qsoftmax_gpu(const torch::Tensor& in, int64_t dim)
{
	// int32_t N = in.numel();
	int32_t N = in.size(1);
	int32_t L = in.size(0);
	const dim3 threads = (64);
    const dim3 blocks((L + threads.x - 1) / threads.x);
	auto out = torch::zeros_like(in);
	const int* x_ptr = in.data_ptr<int>();
	int * y_ptr = out.data_ptr<int>();

	static const int32_t p[] = { 14685058, 114217091, 514075394, 1488269031, 2147475316 };
	static const int32_t p23[] = { 57364 ,446161 ,2008107 , 5813551, 8388575 };

	int32_t *buffer_p;
	int32_t *buffer_p23;

	cudaMalloc((void **)&buffer_p, 5 * sizeof(int32_t));
	cudaMemcpy(buffer_p, p, 5 * sizeof(int32_t), cudaMemcpyHostToDevice);
	cudaMalloc((void **)&buffer_p23, 5 * sizeof(int32_t));
	cudaMemcpy(buffer_p23, p23, 5 * sizeof(int32_t), cudaMemcpyHostToDevice);

	venus_qsoftmax_gpu_kernel<<<blocks, threads>>>(x_ptr, y_ptr, N, L, buffer_p, buffer_p23);
	// venus_qsoftmax_gpu_kernel<<<1, 1>>>(x_ptr, y_ptr, N, L, buffer_p, buffer_p23);
	
	cudaFree(buffer_p);
	cudaFree(buffer_p23);
	
	return out;
}

