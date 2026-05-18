#include <torch/extension.h>

#include <vector>
#include <stdint.h>
#include <math.h>
#include <cmath>

torch::Tensor venusa_qtanh_cpu(torch::Tensor a)
{
	int32_t N = a.numel();
	auto c = torch::empty(a.sizes(), a.options().dtype(torch::kInt64));
	const int *a_ptr = a.data_ptr<int>();
	int64_t *c_ptr = c.data_ptr<int64_t>();

	static const uint32_t bands[] = {0, 33584191, 58182438, 80361293, 102120620, 124483371, 148392654, 174972063, 206183541, 244722657, 281591256, 312045360, 358241226, 403095772, 471425623, 530372454, 2147483648};
	static const uint32_t slopes[] = {2114623134, 1910645334, 1662969581, 1396521636, 1131979061, 881027392, 652221876, 451080022, 281798391, 159724746, 98692663, 60001039, 27632752, 13732196, 2832019, 172634};
	static const uint32_t biass[] = {0,6379959,19800670,39742244,64902158,93996032,125617281,158394439,190900471,218722976,234728779,245973141,256772436,261990874,266776590,268090187,};

	uint32_t i = 0;
	uint32_t j = 0;
	uint32_t sign = 0;
	int64_t absx = 0;
	int64_t slope = 0;
	int64_t bias = 0;
	int64_t tmp = 0;
	int64_t out = 0;

	for (i = 0; i < N; i++)
	{
		tmp = a_ptr[i];

		if (tmp < 0)
		{
			sign = 1;
			if (tmp == -(1LL << 31))
			{
				absx = (1LL << 31) - 1;
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
		if (1 == sign)
		{
			out = (-slope * absx) - (bias << 30);
		}
		else
		{
			out = (slope * absx) + (bias << 30);
		}
		c_ptr[i] = out;
	}

	return c;
}
