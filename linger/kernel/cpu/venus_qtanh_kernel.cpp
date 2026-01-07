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

torch::Tensor venus_qtanh_cpu(torch::Tensor a)
{
	int32_t N = a.numel();
	auto c = torch::zeros_like(a);
	const int *a_ptr = a.data_ptr<int>();
	int *c_ptr = c.data_ptr<int>();

	static const int32_t bands[] = {0, 512, 888, 1226, 1558, 1899, 2264, 2670, 3146, 3734, 4297, 4761, 5466, 6151, 7193, 8093, 32768};
	static const int32_t slopes[] = {32266, 29154, 25374, 21309, 17272, 26886, 19904, 13765, 8599, 19497, 12047, 7324, 3373, 26820, 5531, 337};
	static const int32_t biass[] = {0, 778, 2417, 4851, 7922, 11474, 15334, 19335, 23303, 26699, 28653, 30026, 31344, 31981, 32565, 32725};

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
			if (tmp == -1 * (1 << 15))
			{
				absx = (1 << 15) - 1;
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

				if (j < 6)
				{
					shift = 11;
				}
				else if (j < 10)
				{
					shift = 12;
				}
				else if (j < 14)
				{
					shift = 14;
				}
				else
				{
					shift = 18;
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

		if (1 == sign)
		{
			out = (floor((-1 * slope) * absx * 1.0L / (1 << shift)) - bias);
		}
		else
		{
			out = (floor(slope * absx * 1.0L / (1 << shift)) + bias);
		}
		c_ptr[i] = out;
	}

	return c;
}
