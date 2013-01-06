
#include "backprop.h"

void bp_init(bp * net,
			 int no_of_inputs,
			 int no_of_hiddens,
			 int hidden_layers,
			 int no_of_outputs,
			 unsigned int * random_seed)
{
	int i, j, l;
	bp_neuron * n;

	net->learningRate = 0.2f;
	net->noise = 0.0f;
	net->random_seed = *random_seed;
  
	net->NoOfInputs = no_of_inputs;
	net->inputs = (bp_neuron**)malloc(no_of_inputs*sizeof(bp_neuron*));

	net->NoOfHiddens = no_of_hiddens;
	net->HiddenLayers = hidden_layers;
	net->hiddens =
		(bp_neuron***)malloc(hidden_layers*sizeof(bp_neuron**));
	for (l = 0; l < hidden_layers; l++) {
		net->hiddens[l] =
			(bp_neuron**)malloc(no_of_hiddens*sizeof(bp_neuron*));
	}

	net->NoOfOutputs = no_of_outputs;
	net->outputs = (bp_neuron**)malloc(no_of_outputs*sizeof(bp_neuron*));

	/* create inputs */
	for (i = 0; i < net->NoOfInputs; i++) {
		net->inputs[i] = (bp_neuron*)malloc(sizeof(struct bp_n));
		bp_neuron_init(net->inputs[i], 1, random_seed);
	}

	/* create hiddens */
	for (l = 0; l < hidden_layers; l++) {
		for (i = 0; i < net->NoOfHiddens; i++) {
			net->hiddens[l][i] =
				(bp_neuron*)malloc(sizeof(bp_neuron));
			n = net->hiddens[l][i];
			if (l == 0) {
				bp_neuron_init(n, no_of_inputs, random_seed);
				/* connect to input layer */
				for (j = 0; j < net->NoOfInputs; j++) {
					bp_neuron_add_connection(n, j, net->inputs[j]);
				}
			}
			else {
				bp_neuron_init(n, no_of_hiddens, random_seed);
				/* connect to previous hidden layer */
				for (j = 0; j < net->NoOfHiddens; j++) {
					bp_neuron_add_connection(n, j, net->hiddens[l-1][j]);
				}
			}
		}
	}

	/* create outputs */
	for (i = 0; i < net->NoOfOutputs; i++) {
		net->outputs[i] = (bp_neuron*)malloc(sizeof(bp_neuron));
		n = net->outputs[i];
		bp_neuron_init(n, no_of_hiddens, random_seed);
		for (j = 0; j < net->NoOfHiddens; j++) {
			bp_neuron_add_connection(n, j,
									 net->hiddens[hidden_layers-1][j]);
		}
	}
}

/* deallocates memory */
void bp_free(bp * net)
{
	int l,i;

	for (i = 0; i < net->NoOfInputs; i++) {
		bp_neuron_free(net->inputs[i]);
		free(net->inputs[i]);
		net->inputs[i] = 0;
	}
	free(net->inputs);
	for (l = 0; l < net->HiddenLayers; l++) {
		for (i = 0; i < net->NoOfHiddens; i++) {
			bp_neuron_free(net->hiddens[l][i]);
			free(net->hiddens[l][i]);
			net->hiddens[l][i] = 0;
		}
		free(net->hiddens[l]);
		net->hiddens[l] = 0;
	}
	free(net->hiddens);

	for (i = 0; i < net->NoOfOutputs; i++) {
		bp_neuron_free(net->outputs[i]);
		free(net->outputs[i]);
		net->outputs[i] = 0;
	}
	free(net->outputs);
}

void bp_feed_forward(bp * net)
{  
	int i,l;
	bp_neuron * n;

	for (l = 0; l < net->HiddenLayers; l++) {	
		for (i = 0; i < net->NoOfHiddens; i++) {
			n = net->hiddens[l][i];
			bp_neuron_feedForward(n, net->noise, &net->random_seed);
		}
	}

	for (i = 0; i < net->NoOfOutputs; i++) {
		n = net->outputs[i];
		bp_neuron_feedForward(n, net->noise, &net->random_seed);
	}
}

/* feed forward for a number of layers */
void bp_feed_forward_layers(bp * net, int layers)
{  
	int i,l;
	bp_neuron * n;
	int max = layers;

	if (layers >= net->HiddenLayers) {
		max = net->HiddenLayers;
	}

	for (l = 0; l < max; l++) {	
		for (i = 0; i < net->NoOfHiddens; i++) {
			n = net->hiddens[l][i];
			bp_neuron_feedForward(n, net->noise, &net->random_seed);
		}
	}

	if (layers >= net->HiddenLayers) {
		for (i = 0; i < net->NoOfOutputs; i++) {
			n = net->outputs[i];
			bp_neuron_feedForward(n, net->noise, &net->random_seed);
		}
	}
}


void bp_backprop(bp * net)
{
	int i,l;
	bp_neuron * n;
  
	/* clear all previous backprop errors */
	for (i = 0; i < net->NoOfInputs; i++) {
		n = net->inputs[i];
		n->BPerror = 0;
	}
  
	for (l = 0; l < net->HiddenLayers; l++) {	
		for (i = 0; i < net->NoOfHiddens; i++) {
			n = net->hiddens[l][i];
			n->BPerror = 0;
		}
	}
    
	/* now back-propogate the error from the output units */
	net->BPerrorTotal = 0;
	for (i = 0; i < net->NoOfOutputs; i++) {
		n = net->outputs[i];
		bp_neuron_backprop(n);
		net->BPerrorTotal += n->BPerror;
	}
	net->BPerror = net->BPerrorTotal / net->NoOfOutputs;
  
	for (l = net->HiddenLayers-1; l >= 0; l--) {	
		for (i = 0; i < net->NoOfHiddens; i++) {
			n = net->hiddens[l][i];
			bp_neuron_backprop(n);
			net->BPerrorTotal += n->BPerror;
		}
	}
  
	net->BPerrorTotal /= (net->NoOfOutputs +
						  net->NoOfHiddens); 
}

void bp_learn(bp * net)
{  
	int i,l;
	bp_neuron * n;

	for (l = 0; l < net->HiddenLayers; l++) {	
		for (i = 0; i < net->NoOfHiddens; i++) {
			n = net->hiddens[l][i];
			bp_neuron_learn(n,net->learningRate);
		}
	}

	for (i = 0; i < net->NoOfOutputs; i++) {
		n = net->outputs[i];
		bp_neuron_learn(n,net->learningRate);
	}
}

void bp_set_input(bp * net, int index, float value)
{
	bp_neuron * n;
  
	n = net->inputs[index];
	n->value = value; 
}

static float bp_get_input(bp * net, int index)
{
	bp_neuron * n;
  
	n = net->inputs[index];
	return n->value; 
}

void bp_set_output(bp * net, int index, float value)
{  
	bp_neuron * n;
  
	n = net->outputs[index];
	n->desiredValue = value;
}  

static float bp_get_hidden(bp * net, int layer, int index)
{  
	bp_neuron * n;
  
	n = net->hiddens[layer][index];
	return n->value;
}

float bp_get_output(bp * net, int index)
{  
	bp_neuron * n;
  
	n = net->outputs[index];
	return n->value;
}

void bp_update(bp * net)
{
	bp_feed_forward(net);
    bp_backprop(net);
	bp_learn(net);
}  

static void bp_update_autocoder(bp * net)
{
	int i;

	assert(net->NoOfInputs == net->NoOfOutputs);
	for (i = 0; i < net->NoOfInputs; i++) {
		bp_set_output(net,i,net->inputs[i]->value);
	}

	bp_update(net);
}  

/* coppies the hidden layer from the autocoder to the main network */
void bp_update_from_autocoder(bp * net, bp * autocoder, int hidden_layer)
{
	int i;

	for (i = 0; i < net->NoOfHiddens; i++) {
		bp_neuron_copy(autocoder->hiddens[0][i],
					   net->hiddens[hidden_layer][i]);
	}
}

/* generates an autocoder for the given layer */
void bp_create_autocoder(bp * net, int hidden_layer, bp * autocoder)
{
	int no_of_inputs = net->NoOfHiddens;

	if (hidden_layer==0) no_of_inputs = net->NoOfInputs;

	bp_init(autocoder,
			no_of_inputs,
			net->NoOfHiddens,1,
			no_of_inputs,
			&net->random_seed);
}

/* pre-trains a hidden layer using an autocoder */
void bp_pretrain(bp * net, bp * autocoder, int hidden_layer)
{
	int i;

	/* feed forward to the given hidden layer */
	if (hidden_layer > 0) {
		bp_feed_forward_layers(net, hidden_layer);
	}

	if (hidden_layer > 0) {
		assert(net->NoOfHiddens == autocoder->NoOfInputs);
		for (i = 0; i < net->NoOfHiddens; i++) {
			bp_set_input(autocoder,i,
						 bp_get_hidden(net, hidden_layer, i));
		}
	}
	else {
		assert(autocoder->NoOfInputs==net->NoOfInputs);
		for (i = 0; i < net->NoOfInputs; i++) {
			bp_set_input(autocoder, i,
						 bp_get_input(net, i));
		}
	}
	bp_update_autocoder(autocoder);
}
