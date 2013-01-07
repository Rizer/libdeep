/*
 Face recognition demo
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

#include <stdio.h>
#include "libdeep/globals.h"
#include "libdeep/deeplearn.h"


static void facerec_training()
{
	deeplearn learner;
	int no_of_inputs=10;
	int no_of_hiddens=4;
	int hidden_layers=2;
	int no_of_outputs=2;
	int itt,i;
	unsigned int random_seed = 123;
	float max_backprop_error = 0.03f;
	char filename[256];

	/* create the learner */
	deeplearn_init(&learner,
				   no_of_inputs, no_of_hiddens,
				   hidden_layers,
				   no_of_outputs, &random_seed);

	/* perform pre-training with an autocoder */
	for (itt = 0; itt < 10000; itt++) {
		for (i = 0; i < no_of_inputs; i++) {
			deeplearn_set_input(&learner,i,i/(float)no_of_inputs);
		}
		deeplearn_update(&learner, max_backprop_error);

		if (learner.current_hidden_layer==hidden_layers) {
			break;
		}
	}

	/* perform the final training between the last
	   hidden layer and the outputs */
	for (itt = 0; itt < 10000; itt++) {
		for (i = 0; i < no_of_inputs; i++) {
			deeplearn_set_input(&learner,i,i/(float)no_of_inputs);
		}
		for (i = 0; i < no_of_outputs; i++) {
			deeplearn_set_output(&learner,i,
								 1.0f - (i/(float)no_of_inputs));
		}
		deeplearn_update(&learner, max_backprop_error);

		if (learner.BPerror < max_backprop_error) {
			break;
		}
	}

	/* save a graph */
	sprintf(filename,"%s","training_error.png");
	deeplearn_plot_history(&learner,
						   filename, "Training Error",
						   1024, 480);

	/* free memory */
	deeplearn_free(&learner);
}

int main(int argc, char* argv[])
{	
	facerec_training();
	return 1;
}

