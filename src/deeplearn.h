/*
 libdeep - a library for deep learning
 Copyright (C) 2013  Bob Mottram <bob@robotics.uk.to>

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

#ifndef DEEPLEARN_H
#define DEEPLEARN_H

#include <stdio.h>
#include <stdlib.h>
#include <limits.h>
#include "globals.h"
#include "backprop.h"

struct deepl {
	bp * net;
	bp * autocoder;
	int current_hidden_layer;
	float BPerror;
	unsigned int itterations;
	float * error_threshold;
	int training_complete;

	float history[DEEPLEARN_HISTORY_SIZE];
	int history_index, history_ctr, history_step;
};
typedef struct deepl deeplearn;

void deeplearn_init(deeplearn * learner,
					int no_of_inputs,
					int no_of_hiddens,
					int hidden_layers,
					int no_of_outputs,
					float error_threshold[],
					unsigned int * random_seed);
void deeplearn_feed_forward(deeplearn * learner);
void deeplearn_update(deeplearn * learner);
void deeplearn_free(deeplearn * learner);
void deeplearn_set_input(deeplearn * learner, int index, float value);
void deeplearn_set_output(deeplearn * learner, int index, float value);
float deeplearn_get_output(deeplearn * learner, int index);
int deeplearn_save(FILE * fp, deeplearn * learner);
int deeplearn_load(FILE * fp, deeplearn * learner,
				   unsigned int * random_seed);
int deeplearn_compare(deeplearn * learner1,
					  deeplearn * learner2);
int deeplearn_plot_history(deeplearn * learner,
						   char * filename, char * title,
						   int image_width, int image_height);
void deeplearn_inputs_from_image_patch(deeplearn * learner,
									   unsigned char * img,
									   int image_width, int image_height,
									   int tx, int ty);
void deeplearn_inputs_from_image(deeplearn * learner,
								 unsigned char * img,
								 int image_width, int image_height);
void deeplearn_set_learning_rate(deeplearn * learner, float rate);
void deeplearn_set_dropouts(deeplearn * learner, float dropout_percent);

#endif
