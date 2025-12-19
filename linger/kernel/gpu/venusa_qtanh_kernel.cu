#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <stdint.h>

#define MAX_BITS(bits) ((1LL << (bits - 1)) - 1)
#define MIN_BITS(bits) (-(1LL << (bits - 1)))
#define SATURATE(x, bits)                \
  ((x) > MAX_BITS(bits) ? MAX_BITS(bits) \
						: ((x) < MIN_BITS(bits) ? MIN_BITS(bits) : (x)))
						
__global__ void venusa_qtanh_gpu_kernel(const int* __restrict__ a,
                             int* __restrict__ c, 
                            int32_t len, uint32_t* bands , uint32_t* slopes,
                            uint32_t* biass)
{
    int32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
	uint32_t j = 0;
	uint32_t sign = 0;
	int64_t absx = 0;
	int64_t slope = 0;
	int64_t bias = 0;
	uint32_t shift = 0;
	int64_t tmp = 0;
	int64_t out = 0;
  
    if (idx < len)
    {
        tmp = a[idx];

        if (tmp < 0)
		{
			sign = 1;
			if (tmp == -1 * (1 << 31))
			{
				absx = (1 << 31) - 1;
			}
			else
			{
				absx = -tmp;
			}
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
				bias = biass[j - 1];
				break;
			}
			else
			{
				slope = 0;
				bias = 0;
			}
		}

        bias = bias << 3;
		if (1 == sign)
		{
			out = ((-1 * slope * absx) >> 27) - bias;
		}
		else
		{
			out = ((slope * absx) >> 27) + bias;
		}

		c[idx] = SATURATE(out, 32);
    }
}

torch::Tensor venusa_qtanh_gpu(torch::Tensor a)
{
    int32_t N = a.numel();
    const int threads = 64;
    const dim3 blocks((N + threads - 1) / threads, threads);
    auto c = torch::zeros_like(a);
    const int* a_ptr= a.data_ptr<int>();
    int * c_ptr = c.data_ptr<int>();
	static const uint32_t bands[] = {0, 33584191, 58182438, 80361293, 102120620, 124483371, 148392654, 174972063, 206183541, 244722657, 281591256, 312045360, 358241226, 403095772, 471425623, 530372454, 2147483648};
	static const uint32_t slopes[] = {2114623134, 1910645334, 1662969581, 1396521636, 1131979061, 881027392, 652221876, 451080022, 281798391, 159724746, 98692663, 60001039, 27632752, 13732196, 2832019, 172634};
	static const uint32_t biass[] = {0,6379959,19800670,39742244,64902158,93996032,125617281,158394439,190900471,218722976,234728779,245973141,256772436,261990874,266776590,268090187,};

    uint32_t* buffer_bands;
    uint32_t* buffer_slopes;
    uint32_t* buffer_biass;

    cudaMalloc((void**)&buffer_bands,  17*sizeof(uint32_t));
    cudaMemcpy(buffer_bands, bands, 17*sizeof(uint32_t), cudaMemcpyHostToDevice); 
    cudaMalloc((void**)&buffer_slopes,  16*sizeof(uint32_t));
    cudaMemcpy(buffer_slopes, slopes, 16*sizeof(uint32_t), cudaMemcpyHostToDevice); 
    cudaMalloc((void**)&buffer_biass,  16*sizeof(uint32_t));
    cudaMemcpy(buffer_biass, biass, 16*sizeof(uint32_t), cudaMemcpyHostToDevice); 
    
    venusa_qtanh_gpu_kernel<<<blocks, threads>>>(a_ptr, c_ptr, N, buffer_bands, buffer_slopes, buffer_biass);
    cudaFree(buffer_bands);
    cudaFree(buffer_slopes);
    cudaFree(buffer_biass);

    return c;
}

