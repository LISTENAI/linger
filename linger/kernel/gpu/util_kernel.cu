#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <stdint.h>

#define THREADS_PER_BLOCK 256

#define CUDA_1D_KERNEL_LOOP(i, n)                            \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; \
       i += blockDim.x * gridDim.x)

inline int GET_BLOCKS(const int N) {
	int optimal_block_num = (N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
	int max_block_num = 65000;
	return min(optimal_block_num, max_block_num);
}

__global__ void find_table(float* value, const int32_t* table_index, const float* table, int32_t size){
	CUDA_1D_KERNEL_LOOP(i, size){
		int32_t index = table_index[i];
		value[i] *= table[index];
	}
}

void find_table_gpu(torch::Tensor value, torch::Tensor table, torch::Tensor table_index)
{
    int32_t N = value.numel();
	float* value_ptr = value.data_ptr<float>();
	const float* table_ptr = table.data_ptr<float>();
	const int32_t* table_index_ptr = table_index.data_ptr<int32_t>();
	find_table<<<GET_BLOCKS(THREADS_PER_BLOCK), THREADS_PER_BLOCK>>>(value_ptr, table_index_ptr, table_ptr, N);
}

