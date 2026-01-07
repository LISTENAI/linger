#include <torch/extension.h>

#include <vector>
#include <stdint.h>
#include <math.h>
#include <cmath>
#ifdef INT_MIN

#undef INT_MIN
#undef INT_MAX
#undef UCHAR_MIN
#undef UCHAR_MAX

#define INT_MIN -2147483648
#define INT_MAX 2147483647
#define UCHAR_MIN 0
#define UCHAR_MAX 255

#endif

torch::Tensor venus_qsigmoid_cpu(torch::Tensor a)
{
	int32_t N = a.numel();
	auto c = torch::zeros_like(a);
	const int *a_ptr = a.data_ptr<int>();
	int *c_ptr = c.data_ptr<int>();

	static const int32_t bands[] = {0, 971, 1699, 2347, 2974, 3614, 4291, 5026, 5861, 6831, 7864, 8987, 10375, 11594, 13168, 14730, 32768};
	static const int32_t slopes[] = {32316, 29471, 25891, 22068, 18225, 29057, 22206, 16113, 10881, 27353, 16611, 9242, 4618, 20670, 8610, 693};
	static const int32_t bias0s[] = {16384, 16721, 17464, 18559, 19954, 21585, 23380, 25249, 27121, 28807, 30096, 31107, 31839, 32199, 32502, 32724};
	static const int32_t bias1s[] = {16384, 16047, 15304, 14209, 12814, 11183, 9388, 7519, 5647, 3961, 2672, 1661, 929, 569, 266, 44};

	uint32_t i = 0;
	uint32_t j = 0;
	uint32_t sign = 0;
	int32_t absx = 0;
	int32_t slope = 0;
	int32_t bias = 0;
	int32_t shift = 0;
	int32_t tmp = 0;
	int32_t out = 0;

	for (i = 0; i < N; i++)
	{
		tmp = a_ptr[i];

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
		out = (floor(slope * tmp * 1.0L / (1 << shift)) + bias);
		c_ptr[i] = out;
	}

	return c;
}
