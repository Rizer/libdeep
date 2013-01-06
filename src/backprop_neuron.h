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

#endif
