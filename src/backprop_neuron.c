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

#include "backprop_neuron.h"

/* randomly initialises the weights within the given range */
static void bp_neuron_init_weights(bp_neuron * n,
								   float minVal, float maxVal,
								   unsigned int * random_seed)
{
	float min, max;
	int i;
  
    min = minVal;
    max = maxVal;

	n->min_weight = 9999;
	n->max_weight = -9999;
  
	/* do the weights */
	for (i = 0; i < n->NoOfInputs; i++) {
		n->weights[i] =
			min + (((rand_num(random_seed)%10000)/10000.0f) *
				   (max - min));
		n->lastWeightChange[i] = 0;
		if (n->weights[i] < n->min_weight) {
			n->min_weight = n->weights[i];
		}
		if (n->weights[i] > n->max_weight) {
			n->max_weight = n->weights[i];
		}
	}
  
	/* dont forget the bias value */
	n->bias =
		min + (((rand_num(random_seed)%10000)/10000.0f) *
			   (max - min));
	n->lastBiasChange = 0; 
}

/* copy weights from one neuron to another */
void bp_neuron_copy(bp_neuron * source,
					bp_neuron * dest)
{
	int i;

	if (source->NoOfInputs !=
		dest->NoOfInputs) {
		printf("Warning: neurons have different numbers of inputs\n");
		return;
	}

	for (i = 0; i < source->NoOfInputs; i++) {
		dest->weights[i] = source->weights[i];
	}
	dest->bias = source->bias;
}

/* initialises the neuron */
void bp_neuron_init(bp_neuron * n,
					int no_of_inputs,
					unsigned int * random_seed)
{
	int i;

	assert(no_of_inputs > 0);
	n->NoOfInputs = no_of_inputs;
	n->weights = (float*)malloc(no_of_inputs*sizeof(float));
	n->lastWeightChange = (float*)malloc(no_of_inputs*sizeof(float));
	n->inputs = (struct bp_n **)malloc(no_of_inputs*
									   sizeof(struct bp_n *));
	bp_neuron_init_weights(n, -0.1f, 0.1f, random_seed);
	n->desiredValue = -1;
	n->value = 0;
	n->BPerror = 0;
	n->excluded = 0;

	/* clear the input pointers */
	for (i = 0; i < no_of_inputs; i++) {
		n->inputs[i] = 0;		
	}
}

/* compares two neurons and returns a non-zero value
   if they are the same */
int bp_neuron_compare(bp_neuron * n1, bp_neuron * n2)
{
	int i;

	if ((n1->NoOfInputs != n2->NoOfInputs) ||
		(n1->bias != n2->bias)) {
		return 0;
	}

	for (i = 0; i < n1->NoOfInputs; i++) {
		if ((n1->weights[i] != n2->weights[i]) ||
			(n1->lastWeightChange[i] != n2->lastWeightChange[i])) {
			return 0;
		}
	}
	return 1;
}

/* free memory */
void bp_neuron_free(bp_neuron * n)
{
	int i;

	free(n->weights);
	for (i = 0; i < n->NoOfInputs; i++) {
		n->inputs[i]=0;
	}
	free(n->inputs);
	free(n->lastWeightChange);
}

/* activation function */
static float af(float x)
{
	return x * (1.0f - x);
}


/* adds a connection to a neuron */
void bp_neuron_add_connection(bp_neuron * dest,
							  int index, bp_neuron * source)
{
	dest->inputs[index] = source;
}

/* feed forward */
void bp_neuron_feedForward(bp_neuron * n,
						   float noise,
						   unsigned int * random_seed)
{
	float adder;
	int i;
  
	if (n->excluded > 0) return;

	adder = n->bias;

	for (i = 0; i < n->NoOfInputs; i++) {
		if (n->inputs[i] != 0) {
			adder += n->weights[i] * n->inputs[i]->value;
		}
	}
  
	/* add some random noise */
	if (noise > 0) {
		adder = ((1.0f - noise) * adder) +
			(noise * ((rand_num(random_seed)%10000)/10000.0f));
	}
  
	n->value = 1.0f / (1.0f + exp(-adder));
}

/* back-propagate the error */
void bp_neuron_backprop(bp_neuron * n)
{
	int i;
	bp_neuron * nrn;
	float afact;

	if (n->excluded > 0) return;
  
	if (n->desiredValue > -1) {
		/* output unit */
		n->BPerror = n->desiredValue - n->value;
	}
  
	afact = af(n->value);
  
	for (i = 0; i < n->NoOfInputs; i++) {
		nrn = n->inputs[i];
		if (nrn != 0) {
			nrn->BPerror +=
				(n->BPerror * afact * n->weights[i]);
		}
	}
}


/* adjust the weights */
void bp_neuron_learn(bp_neuron * n,
					 float learningRate)
{
	int i;
	float afact,e,gradient;

	if (n->excluded > 0) return;

	e = learningRate / (1.0f + n->NoOfInputs);
	afact = af(n->value);
	gradient = afact * n->BPerror;
	n->lastBiasChange = e * (n->lastBiasChange + 1.0f) * gradient;
	n->bias += n->lastBiasChange;
	n->min_weight = 9999;
	n->max_weight = -9999;
	for (i = 0; i < n->NoOfInputs; i++) {
		if (n->inputs[i] != 0) {
			n->lastWeightChange[i] =
				e * (n->lastWeightChange[i] + 1) *
				gradient * n->inputs[i]->value;
			n->weights[i] += n->lastWeightChange[i];
			if (n->weights[i] < n->min_weight) {
				n->min_weight = n->weights[i];
			}
			if (n->weights[i] > n->max_weight) {
				n->max_weight = n->weights[i];
			}
		}
	}
}

/* saves neuron parameters.  Note that there is no need to
   save the connections, since layers are always fully
   interconnected */
int bp_neuron_save(FILE * fp, bp_neuron * n)
{
	int retval;
	
	retval = fwrite(&n->NoOfInputs, sizeof(int), 1, fp);

	retval = fwrite(n->weights, sizeof(float),
					n->NoOfInputs, fp);
	retval = fwrite(n->lastWeightChange, sizeof(float),
					n->NoOfInputs, fp);

	retval = fwrite(&n->min_weight, sizeof(float), 1, fp);
	retval = fwrite(&n->max_weight, sizeof(float), 1, fp);

	retval = fwrite(&n->bias, sizeof(float), 1, fp);
	retval = fwrite(&n->lastBiasChange, sizeof(float), 1, fp);
	retval = fwrite(&n->desiredValue, sizeof(float), 1, fp);

	return retval;
}

/* load neuron parameters */
int bp_neuron_load(FILE * fp, bp_neuron * n)
{
	int retval;

	retval = fread(&n->NoOfInputs, sizeof(int), 1, fp);

	retval = fread(n->weights, sizeof(float),
				   n->NoOfInputs, fp);
	retval = fread(n->lastWeightChange, sizeof(float),
				   n->NoOfInputs, fp);

	retval = fread(&n->min_weight, sizeof(float), 1, fp);
	retval = fread(&n->max_weight, sizeof(float), 1, fp);

	retval = fread(&n->bias, sizeof(float), 1, fp);
	retval = fread(&n->lastBiasChange, sizeof(float), 1, fp);
	retval = fread(&n->desiredValue, sizeof(float), 1, fp);

	n->value = 0;
	n->BPerror = 0;
	n->excluded = 0;

	return retval;
}
