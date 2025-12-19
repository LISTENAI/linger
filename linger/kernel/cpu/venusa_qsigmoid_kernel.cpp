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

torch::Tensor venusa_qsigmoid_cpu(torch::Tensor a)
{
	int32_t N = a.numel();
	auto c = torch::zeros_like(a);
	const int *a_ptr = a.data_ptr<int>();
	int *c_ptr = c.data_ptr<int>();

	static const uint32_t bands[] = {0, 63656107, 111395083, 153816608, 194953701, 236865036, 281253298, 329433241, 384154475, 447714743, 515399149, 589016819, 679940425, 759830874, 862986200, 965402453, 2147483648};
	static const uint32_t slopes[] = {529475578, 482862538, 424212188, 361565531, 298613704, 238039992, 181912633, 132002182, 89144402, 56020914, 34019888, 18928477, 9459126, 5291645, 2204341, 177654};
	static const uint32_t bias0s[] = {134217728,136981152,143065818,152040132,163469967,176832392,191534255,206847211,222180513,235991910,246552465,254831080,260827489,263776596,266257920,268080117};
	static const uint32_t bias1s[] = {134217728,131454303,125369637,116395323,104965488,91603063,76901200,61588244,46254942,32443545,21882990,13604375,7607967,4658859,2177535,355339};

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
			}
		}
        bias = bias << 3;
		out = ((slope * tmp) >> 27) + bias;
		c_ptr[i] = SATURATE(out, 32);
	}

	return c;
}
