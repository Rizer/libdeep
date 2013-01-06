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

/* returns teh value of an output */
float deeplearn_get_output(deeplearn * learner, int index)
{
	return bp_get_output(learner->net, index);
}

