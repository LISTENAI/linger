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

static int32_t shift_pure(int64_t v, int32_t s)
{
    if (s >= 63 || s <= -63) {
		return 0;
	}
	if (s > 0) {
		v = v << s;
	} else {
        v = v >> (-s);
    }
    return SATURATE(v, 32);
}

static int32_t sub32s(int32_t a, int32_t b)
{
	int64_t s = 0;
	s = (int64_t)((int64_t)a - (int64_t)b);
	return SATURATE(s, 32);
}
static int32_t add32s(int32_t a, int32_t b)
{
	int64_t s = 0;
	s = (int64_t)((int64_t)a + (int64_t)b);
	return SATURATE(s, 32);
}

#if 0
static int32_t shift_rasym(int64_t x,int32_t n)
{
	// double y = (double)x * (double)pow(2, n);
	// y = floor(y + 0.5);
	// return (int32_t)y;

    if (n >= 63 || n <= -63) {
		return 0;
	}
    if (n >= 0) {
		x = x << n;
	} 
	else {
        n = (-n);
        x = x >> (n - 1);
		x = (x & 0x1) + (x >> 1);
    }
    return x;
}

static int32_t shift_rasyms(int64_t x, int32_t n)
{
	// double y = (double)x * (double)pow(2, n);
	// y = floor(y + 0.5);
	// return SATURATE(y, 32);
	if (n >= 64 || n <= -64) {
		return 0;
	}
    if (n >= 0) {
		x = x << n;
	} 
	else {
        n = (-n);
        x = x >> (n - 1);
		x = (x & 0x1) + (x >> 1);
    }
	return SATURATE(x, 32);
}
#endif

static int64_t shfit_floor_x05_int64(int64_t x, int32_t shift)
{
	int64_t val = x;

	if (shift >= 64) {
		return 0;
	}
	if (shift > 0) {
		val = val >> (shift - 1);
		val = (val & 0x1) + (val >> 1);
	}

	return val;
}

/*-------------------------------------------------------------------------
Softmax
The function computes the softmax (normalized exponential function) of
input data. 32-bit fixed-point functions accept inputs in Q6.25 and form
outputs in Q16.15 format.

Precision:
32x32  32-bit inputs, 32-bit output. Accuracy: 2 LSB (see Note below)
f      floating point input, floating point output

Note: Accuracy of function may depend on amount of data and their
distribution. Given accuracy is achieved for N=2 for any pair of data
from input domain.

Input:
in.data_ptr   input data, Q6.25
in.numel()    length of vectors
Output:
out.data_ptr  result, Q16.15 or floating point

Restriction:
in,out should not overlap
-------------------------------------------------------------------------*/
torch::Tensor venusa_qsoftmax_cpu(const torch::Tensor& in, int64_t dim)
{
	// int32_t N = in.numel();
	int32_t N = in.size(1);
	int32_t L = in.size(0);
	auto out = torch::zeros_like(in);
	int *p_x = in.data_ptr<int>();
	int *p_y = out.data_ptr<int>();

	const static int32_t p23[5] = { 57364 ,446161 ,2008107 , 5813551, 8388575 };

	for (int k = 0; k < L; k++)
	{
		uint32_t A = 0x800000, B = 0, C = 0;
		int *x_ptr = p_x + N * k;
		int *y_ptr = p_y + N * k;

		int32_t max_value = x_ptr[0];
		int32_t data = 0;
		int32_t X = 0;
		int64_t Y = 0;
		int32_t E = 0;
		int64_t E_SUM = 0;
		
		for (int i = 1; i < N; i++)
		{
			max_value = x_ptr[i] > max_value ? x_ptr[i] : max_value;
		}
        if (max_value == (int32_t)0x80000000)
            max_value += 1;

		for (int i = 0; i < N; i++)
		{
			data = sub32s(x_ptr[i], max_value);
            X = shfit_floor_x05_int64((int64_t)X * (int64_t)774541002, 31);
			// X = shift_rasyms((int64_t)data * (int64_t)774541002, -31);//exp=>2xp，Q6.25=>Q8.23
			E = X >> 23;
			E = E + 1;//与118行对应

			X = X & 0x7fffff;
			X = X - 0x800000;

			Y = p23[0];
            for (int j = 1; j < 5; j++) 
            {
                int64_t t = (((int64_t)Y * (int64_t)X) <<7) + ((int64_t)p23[j] << 30);
                if (j < 4) {
                    Y = shfit_floor_x05_int64(t, 30);
                } else {
                    if (30 - E > 63) {
                        Y = 0;
                    } else {
                        Y = shfit_floor_x05_int64(t, 30 - E);
                    }
                }
            } 
            ((int32_t*)y_ptr)[i] = Y;
		}

		for (int i = 0; i < N; i++)
		{
			E_SUM += y_ptr[i];
		}

		B = SATURATE(E_SUM, 32);
		for (int i = 1; i <= 30; i++) {
			if(A>=B) {
				C = C+1; 
				C = C*2;
				A = A-B;
				A = A*2;
			} else {
				C = C*2;
				A = A*2;
			}
		}

		for (int i = 0; i < N; i++)
		{
			y_ptr[i] = shfit_floor_x05_int64((int64_t)C * (int64_t)y_ptr[i], 38);
		}
	}

	return out;
}

