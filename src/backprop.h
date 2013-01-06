#ifndef BACKPROP_H
#define BACKPROP_H

#include <stdio.h>
#include <stdlib.h>
#include "backprop_neuron.h"

struct backprop {
	int NoOfInputs,NoOfHiddens,NoOfOutputs;
	int HiddenLayers;
	bp_neuron ** inputs;
	bp_neuron *** hiddens;
	bp_neuron ** outputs;
	float BPerrorTotal;
	float BPerror;
	float learningRate;
	float noise;
	unsigned int random_seed;
};
typedef struct backprop bp;

void bp_init(bp * net,
			 int no_of_inputs,
			 int no_of_hiddens,
			 int hidden_layers,
			 int no_of_outputs,
			 unsigned int * random_seed);
void bp_free(bp * net);
void bp_feed_forward(bp * net);
void bp_feed_forward_layers(bp * net, int layers);
void bp_backprop(bp * net);
void bp_learn(bp * net);
void bp_set_input(bp * net, int index, float value);
void bp_set_output(bp * net, int index, float value);
float bp_get_output(bp * net, int index);
void bp_update(bp * net);
void bp_create_autocoder(bp * net, int hidden_layer, bp * autocoder);
void bp_pretrain(bp * net, bp * autocoder, int hidden_layer);
void bp_update_from_autocoder(bp * net, bp * autocoder, int hidden_layer);

#endif
