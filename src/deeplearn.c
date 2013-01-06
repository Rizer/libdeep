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

#include "deeplearn.h"

/* initialise a deep learner */
void deeplearn_init(deeplearn * learner,
					int no_of_inputs,
					int no_of_hiddens,
					int hidden_layers,
					int no_of_outputs,
					unsigned int * random_seed)
{
	/* set the current layer being trained */
	learner->current_hidden_layer = 0;
	
	/* create the network */
	learner->net = (bp*)malloc(sizeof(bp));

	/* initialise the network */
	bp_init(learner->net,
			no_of_inputs, no_of_hiddens,
			hidden_layers, no_of_outputs,
			random_seed);

	/* create the autocoder */
	learner->autocoder = (bp*)malloc(sizeof(bp));
	bp_create_autocoder(learner->net,
						learner->current_hidden_layer,
						learner->autocoder);

	learner->BPerror = 9999;
}

void deeplearn_feed_forward(deeplearn * learner)
{
	bp_feed_forward(learner->net);
}

void deeplearn_update(deeplearn * learner,
					  float max_backprop_error)
{
	/* pretraining */
	if (learner->current_hidden_layer <
		learner->net->HiddenLayers) {

		/* train the autocoder */
		bp_pretrain(learner->net, learner->autocoder,
					learner->current_hidden_layer);

		/* update the backprop error value */
		learner->BPerror = learner->autocoder->BPerror;

		/* if below the errro threshold */
		if (learner->net->BPerror < max_backprop_error) {
			/* copy the hidden units */
			bp_update_from_autocoder(learner->net,
									 learner->autocoder,
									 learner->current_hidden_layer);

			/* delete the existing autocoder */
			bp_free(learner->autocoder);
			free(learner->autocoder);
			learner->autocoder = 0;

			/* advance to the next hidden layer */
			learner->current_hidden_layer++;

			/* if not the final hidden layer */
			if (learner->current_hidden_layer <
				learner->net->HiddenLayers) {

				/* make a new autocoder */
				learner->autocoder = (bp*)malloc(sizeof(bp));
				bp_create_autocoder(learner->net,
									learner->current_hidden_layer,
									learner->autocoder);
			}
		}
	}
	else {
		/* ordinary network training */
		bp_update(learner->net);

		/* update the backprop error value */
		learner->BPerror = learner->net->BPerror;
	}
}

/* free the deep learner's allocated memory */
void deeplearn_free(deeplearn * learner)
{
	bp_free(learner->net);
	free(learner->net);
	if (learner->autocoder != 0) {
		bp_free(learner->autocoder);
		free(learner->autocoder);
	}
}

/* sets the value of an input to the network */
void deeplearn_set_input(deeplearn * learner, int index, float value)
{
	bp_set_input(learner->net, index, value);
}

/* sets the value of an output */
void deeplearn_set_output(deeplearn * learner, int index, float value)
{
	bp_set_output(learner->net, index, value);
}

/* returns the value of an output */
float deeplearn_get_output(deeplearn * learner, int index)
{
	return bp_get_output(learner->net, index);
}

/* save to file */
int deeplearn_save(FILE * fp, deeplearn * learner)
{
	int retval,val;

	retval = fwrite(&learner->current_hidden_layer, sizeof(int), 1, fp);
	retval = fwrite(&learner->BPerror, sizeof(float), 1, fp);

	retval = bp_save(fp, learner->net);
	if (learner->autocoder != 0) {
		val = 1;
		retval = fwrite(&val, sizeof(int), 1, fp);
		retval = bp_save(fp, learner->autocoder);
	}
	else {
		val = 0;
		retval = fwrite(&val, sizeof(int), 1, fp);
	}
	return retval;
}

/* load from file */
int deeplearn_load(FILE * fp, deeplearn * learner,
				   unsigned int * random_seed)
{
	int retval,val=0;

	retval = fread(&learner->current_hidden_layer, sizeof(int), 1, fp);
	retval = fread(&learner->BPerror, sizeof(float), 1, fp);

	learner->net = (bp*)malloc(sizeof(bp));
	retval = bp_load(fp, learner->net, random_seed);
	retval = fread(&val, sizeof(int), 1, fp);
	if (val == 1) {
		learner->autocoder = (bp*)malloc(sizeof(bp));
		retval = bp_load(fp, learner->autocoder, random_seed);
	}
	else {
		learner->autocoder = 0;
	}
	return retval;
}

/* compares two deep learners and returns a greater
   than zero value if they are the same */
int deeplearn_compare(deeplearn * learner1,
					  deeplearn * learner2)
{
	int retval;

	if (learner1->current_hidden_layer !=
		learner2->current_hidden_layer) {
		return -1;
	}
	if (learner1->BPerror != learner2->BPerror) {
		return -2;
	}
	retval = bp_compare(learner1->net,learner2->net);
	if (retval < 1) return -3;
	if ((learner1->autocoder==0) !=
		(learner2->autocoder==0)) {
		return -4;
	}
	return 1;
}
