/*
 libdeep - a library for deep learning
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

#ifndef BACKPROP_NEURON_H
#define BACKPROP_NEURON_H

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <assert.h>
#include <ctype.h>
#include <math.h>
#include <assert.h>
#include <time.h>
#include <omp.h>
#include "deeplearn_random.h"

struct bp_n {
	int NoOfInputs;
	float * weights;
	float * lastWeightChange;
	struct bp_n ** inputs;
	float bias;
	float lastBiasChange;
	float BPerror;

	float value;
	float desiredValue;
};
typedef struct bp_n bp_neuron;

void bp_neuron_init(bp_neuron * n,
					int no_of_inputs,
					unsigned int * random_seed);
void bp_neuron_add_connection(bp_neuron * dest,
							  int index, bp_neuron * source);
void bp_neuron_feedForward(bp_neuron * n,
						   float noise,
						   unsigned int * random_seed);
void bp_neuron_backprop(bp_neuron * n);
void bp_neuron_learn(bp_neuron * n,
					 float learningRate);
void bp_neuron_free(bp_neuron * n);
void bp_neuron_copy(bp_neuron * source,
					bp_neuron * dest);
int bp_neuron_save(FILE * fp, bp_neuron * n);
int bp_neuron_load(FILE * fp, bp_neuron * n);
int bp_neuron_compare(bp_neuron * n1, bp_neuron * n2);

#endif
