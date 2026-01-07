#ifdef INT_MIN

#undef INT_MIN
#undef INT_MAX
#undef UCHAR_MIN
#undef UCHAR_MAX

#define INT_MIN    -2147483648
#define INT_MAX    2147483647
#define UCHAR_MIN  0
#define UCHAR_MAX  255

#endif
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <stdint.h>


__global__ void venus_qsigmoid_gpu_kernel(const int* __restrict__ a,
                             int* __restrict__ c, 
                            int32_t len, int32_t* bands , int32_t* slopes,
                            int32_t* bias0s , int32_t* bias1s)
{
    int32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t j = 0;
    uint32_t sign = 0;
    uint32_t absx = 0;
    int32_t slope = 0;
    int32_t bias = 0;
    int32_t shift = 0;
    int32_t tmp = 0;
    int32_t out = 0;
	
    if (idx < len)
    {
		tmp = a[idx];

        if (tmp < 0)
		{
			sign = 1;
			absx = -tmp;
		}
		else
		{
			sign = 0;
			absx = tmp;
		}


		for (j = 1; j < 17; ++j)
		{

			if (absx <= bands[j])
			{
				slope = slopes[j - 1];
				if (1 == sign)
				{
					bias = bias1s[j - 1];
				}
				else
				{
					bias = bias0s[j - 1];
				}

				if (j < 6)
				{
					shift = 13;
				}
				else if (j < 10)
				{
					shift = 14;
				}
				else if (j < 14)
				{
					shift = 16;
				}
				else
				{
					shift = 19;
				}
				break;
			}
			else
			{
				slope = 0;
				bias = 0;
				shift = 0;
			}
		}

		out = (floor(slope * tmp * 1.0f / (1 << shift)) + bias);
		c[idx] = out;
    }
}

torch::Tensor venus_qsigmoid_gpu(torch::Tensor a)
{
    int32_t N = a.numel();
    const int threads = 64;
    const dim3 blocks((N + threads - 1) / threads, threads);
    auto c = torch::zeros_like(a);
    const int* a_ptr= a.data_ptr<int>();
    int * c_ptr = c.data_ptr<int>();
	static const int32_t bands[]  = {0, 971, 1699, 2347, 2974, 3614, 4291, 5026, 5861, 6831, 7864, 8987, 10375, 11594, 13168, 14730, 32768};
	static const int32_t slopes[] = {32316, 29471, 25891, 22068, 18225, 29057, 22206, 16113, 10881, 27353, 16611, 9242,  4618,  20670, 8610,  693};
	static const int32_t bias0s[] = {16384, 16721, 17464, 18559, 19954, 21585, 23380, 25249, 27121, 28807, 30096, 31107, 31839, 32199, 32502, 32724};
	static const int32_t bias1s[] = {16384, 16047, 15304, 14209, 12814, 11183, 9388,  7519,  5647,  3961,  2672,  1661,  929,   569,   266,   44   };
        
    int32_t* buffer_bands;
    int32_t* buffer_slopes;
    int32_t* buffer_bias0s;
	int32_t* buffer_bias1s;


    cudaMalloc((void**)&buffer_bands,  17*sizeof(int32_t));
    cudaMemcpy(buffer_bands, bands, 17*sizeof(int32_t), cudaMemcpyHostToDevice); 
    cudaMalloc((void**)&buffer_slopes,  16*sizeof(int32_t));
    cudaMemcpy(buffer_slopes, slopes, 16*sizeof(int32_t), cudaMemcpyHostToDevice); 
    cudaMalloc((void**)&buffer_bias0s,  16*sizeof(int32_t));
    cudaMemcpy(buffer_bias0s, bias0s, 16*sizeof(int32_t), cudaMemcpyHostToDevice); 
    cudaMalloc((void**)&buffer_bias1s,  16*sizeof(int32_t));
    cudaMemcpy(buffer_bias1s, bias1s, 16*sizeof(int32_t), cudaMemcpyHostToDevice); 
    
    venus_qsigmoid_gpu_kernel<<<blocks, threads>>>(a_ptr, c_ptr, N, buffer_bands, buffer_slopes, buffer_bias0s, buffer_bias1s );
    cudaFree(buffer_bands);
    cudaFree(buffer_slopes);
    cudaFree(buffer_bias0s);
	cudaFree(buffer_bias1s);

    return c;
}

