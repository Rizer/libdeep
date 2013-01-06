#include "deeplearn_random.h"

/* Lehmer random number generator */
int rand_num(unsigned int * seed)
{
	unsigned int v =
		((unsigned long long)(*seed) * 279470273UL) % 4294967291UL;
	if (v==0) v = (int)time(NULL); /* avoid the singularity */
	*seed = v;
	return abs((int)v);
}
