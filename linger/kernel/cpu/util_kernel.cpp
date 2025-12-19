#include <torch/extension.h>

#include <vector>
#include <stdint.h>
#include <math.h>

namespace extension_util_cpp
{
	void find_table(float *value, const int32_t *table_index, const float *table, int32_t size)
	{
		for (int i = 0; i < size; i++)
		{
			int32_t index = table_index[i];
			value[i] *= table[index];
		}
	}
} // namespace extension_util_cpp

void find_table_cpu(torch::Tensor value, torch::Tensor table, torch::Tensor table_index)
{
	int32_t N = value.numel();
	float *value_ptr = value.data_ptr<float>();
	const float *table_ptr = table.data_ptr<float>();
	const int32_t *table_index_ptr = table_index.data_ptr<int32_t>();
	extension_util_cpp::find_table(value_ptr, table_index_ptr, table_ptr, N);
}
