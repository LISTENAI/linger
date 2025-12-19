#include <torch/extension.h>

#include <vector>
#include <stdint.h>
#include <math.h>
#include <cmath>

#define MAX_BITS(bits) ((1LL << (bits - 1)) - 1)
#define MIN_BITS(bits) (-(1LL << (bits - 1)))
#define SATURATE(x, bits)                \
  ((x) > MAX_BITS(bits) ? MAX_BITS(bits) \
                        : ((x) < MIN_BITS(bits) ? MIN_BITS(bits) : (x)))

torch::Tensor arcs_qtanh_cpu(torch::Tensor a)
{
	int32_t N = a.numel();
	auto c = torch::zeros_like(a);
	const int *a_ptr = a.data_ptr<int>();
	int *c_ptr = c.data_ptr<int>();

	static const uint32_t bands[] = {0, 33584191, 58182438, 80361293, 102120620, 124483371, 148392654, 174972063, 206183541, 244722657, 281591256, 312045360, 358241226, 403095772, 471425623, 530372454, 2147483648};
	static const uint32_t slopes[] = {2114623134, 1910645334, 1662969581, 1396521636, 1131979061, 881027392, 652221876, 451080022, 281798391, 159724746, 98692663, 60001039, 27632752, 13732196, 2832019, 172634};
	static const uint32_t biass[] = {0, 51039676, 158405367, 317937954, 519217264, 751968259, 1004938252, 1267155516, 1527203771, 1749783808, 1877830237, 1967785133, 2054179492, 2095926998, 2134212723, 2144721503};

	uint32_t i = 0;
	uint32_t j = 0;
	uint32_t sign = 0;
	int64_t absx = 0;
	int64_t slope = 0;
	int64_t bias = 0;
	int32_t shift = 0;
	int64_t tmp = 0;
	int64_t out = 0;

	for (i = 0; i < N; i++)
	{
		tmp = a_ptr[i];

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

		if (1 == sign)
		{
			out = ((-1 * slope * absx) >> 27) - bias;
		}
		else
		{
			out = ((slope * absx) >> 27) + bias;
		}

		c_ptr[i] = SATURATE(out, 32);
	}

	return c;
}
