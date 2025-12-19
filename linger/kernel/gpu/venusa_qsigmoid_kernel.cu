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

__global__ void venusa_qsigmoid_gpu_kernel(const int* __restrict__ a,
                             int* __restrict__ c, 
                            int32_t len, uint32_t* bands , uint32_t* slopes,
                            uint32_t* bias0s , uint32_t* bias1s)
{
    int32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t j = 0;
    uint32_t sign = 0;
    int64_t absx = 0;
    int64_t slope = 0;
    int64_t bias = 0;
    int32_t shift = 0;
    int64_t tmp = 0;
    int64_t out = 0;
	
    if (idx < len)
    {
		tmp = a[idx];

        if (tmp < 0)
		{
			sign = 1;
			if (tmp == (-1ULL<<31)) {
                absx = (1ULL<<31)-1;
            } else {
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
				if (1 == sign)
				{
					bias = bias1s[j - 1];
				}
				else
				{
					bias = bias0s[j - 1];
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

        bias = bias << 3;
		out = ((slope * tmp) >> 27) + bias;
		c[idx] = SATURATE(out, 32);
    }
}

torch::Tensor venusa_qsigmoid_gpu(torch::Tensor a)
{
    int32_t N = a.numel();
    const int threads = 64;
    const dim3 blocks((N + threads - 1) / threads, threads);
    auto c = torch::zeros_like(a);
    const int* a_ptr= a.data_ptr<int>();
    int * c_ptr = c.data_ptr<int>();
	static const uint32_t bands[] = {0, 63656107, 111395083, 153816608, 194953701, 236865036, 281253298, 329433241, 384154475, 447714743, 515399149, 589016819, 679940425, 759830874, 862986200, 965402453, 2147483648};
	static const uint32_t slopes[] = {529475578, 482862538, 424212188, 361565531, 298613704, 238039992, 181912633, 132002182, 89144402, 56020914, 34019888, 18928477, 9459126, 5291645, 2204341, 177654};
	static const uint32_t bias0s[] = {134217728,136981152,143065818,152040132,163469967,176832392,191534255,206847211,222180513,235991910,246552465,254831080,260827489,263776596,266257920,268080117};
	static const uint32_t bias1s[] = {134217728,131454303,125369637,116395323,104965488,91603063,76901200,61588244,46254942,32443545,21882990,13604375,7607967,4658859,2177535,355339};

    uint32_t* buffer_bands;
    uint32_t* buffer_slopes;
    uint32_t* buffer_bias0s;
	uint32_t* buffer_bias1s;


    cudaMalloc((void**)&buffer_bands,  17*sizeof(uint32_t));
    cudaMemcpy(buffer_bands, bands, 17*sizeof(uint32_t), cudaMemcpyHostToDevice); 
    cudaMalloc((void**)&buffer_slopes,  16*sizeof(uint32_t));
    cudaMemcpy(buffer_slopes, slopes, 16*sizeof(uint32_t), cudaMemcpyHostToDevice); 
    cudaMalloc((void**)&buffer_bias0s,  16*sizeof(uint32_t));
    cudaMemcpy(buffer_bias0s, bias0s, 16*sizeof(uint32_t), cudaMemcpyHostToDevice); 
    cudaMalloc((void**)&buffer_bias1s,  16*sizeof(uint32_t));
    cudaMemcpy(buffer_bias1s, bias1s, 16*sizeof(uint32_t), cudaMemcpyHostToDevice); 
    
    venusa_qsigmoid_gpu_kernel<<<blocks, threads>>>(a_ptr, c_ptr, N, buffer_bands, buffer_slopes, buffer_bias0s, buffer_bias1s );
    cudaFree(buffer_bands);
    cudaFree(buffer_slopes);
    cudaFree(buffer_bias0s);
	cudaFree(buffer_bias1s);

    return c;
}

