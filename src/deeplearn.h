#ifndef DEEP_H
#define DEEP_H

#include <stdio.h>
#include <stdlib.h>
#include "backprop.h"

struct deepl {
	bp * net;
	bp * autocoder;
	int current_hidden_layer;
	float BPerror;
};
typedef struct deepl deeplearn;

void deeplearn_init(deeplearn * learner,
					int no_of_inputs,
					int no_of_hiddens,
					int hidden_layers,
					int no_of_outputs,
					unsigned int * random_seed);
void deeplearn_update(deeplearn * learner,
					  float max_backprop_error);
void deeplearn_free(deeplearn * learner);
void deeplearn_set_input(deeplearn * learner, int index, float value);
void deeplearn_set_output(deeplearn * learner, int index, float value);
float deeplearn_get_output(deeplearn * learner, int index);

#endif
