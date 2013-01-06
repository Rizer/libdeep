/*
 Copyright (C) 2013  Bob Mottram <bob@sluggish.dyndns.org>

 Redistribution and use in source and binary forms, with or without
 modification, are permitted provided that the following conditions
 are met:
 1. Redistributions of source code must retain the above copyright
    notice, this list of conditions and the following disclaimer.
 2. Redistributions in binary form must reproduce the above copyright
    notice, this list of conditions and the following disclaimer in the
    documentation and/or other materials provided with the distribution.
 3. Neither the name of the University nor the names of its contributors
    may be used to endorse or promote products derived from this software
    without specific prior written permission.
 .
 THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 ``AS IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT 
 LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR 
 A PARTICULAR PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE HOLDERS OR
 CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, 
 EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, 
 PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR 
 PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF 
 LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING 
 NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS 
 SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#include "tests_random.h"

static void test_rand_num()
{
	unsigned int random_seed=0;
	int i,j,t,v,min=0,max=0;
	int test1[50];
	int test2[50];
	int test3[50];
	int * result;
	int same=0,repeats=0;

	printf("test_rand_num...");

	/* run three sequences with different seeds */
	for (t = 0; t < 3; t++) {
		switch(t) {
		case 0: { random_seed = 123; result = (int*)test1; break; }
		case 1: { random_seed = 555; result = (int*)test2; break; }
		case 2: { random_seed = 8323; result = (int*)test3; break; }
		}
		for (i = 0; i < 50; i++) {
			result[i] = rand_num(&random_seed);
		}
	}

	for (i = 0; i < 50; i++) {
		/* check that the sequences are different */
		if ((test1[i]==test2[i]) ||
			(test1[i]==test3[i]) ||
			(test2[i]==test3[i])) {
			same++;
		}

		/* check the number of repeats within each sequence */
		for (j = 0; j < 50; j++) {
			if (i!=j) {
				if ((test1[i]==test1[j]) ||
					(test2[i]==test2[j]) ||
					(test3[i]==test3[j])) {
					repeats++;
				}
			}
		}
	}		
	assert(same < 2);
	assert(repeats < 2);

	/* check that the range is not too restricted */
	for (i = 0; i < 10000; i ++) {
		v = rand_num(&random_seed);
		if ((i==0) ||
			((i>0) && (v<min))) {
			min = v;
		}
		if ((i==0) ||
			((i>0) && (v>max))) {
			max = v;
		}
	}
	assert(max > min);
	assert(min >= 0);
	assert(max - min > 60000);

	printf("Ok\n");
}

int run_tests_random()
{
	printf("\nRunning random number generator tests\n");

	test_rand_num();

	printf("All random number generator tests completed\n");
	return 1;
}
