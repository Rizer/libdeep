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
	net->BPerror = DEEPLEARN_UNKNOWN_ERROR;
	net->BPerrorAverage = DEEPLEARN_UNKNOWN_ERROR;
	net->BPerrorTotal = DEEPLEARN_UNKNOWN_ERROR;
	net->itterations = 0;
	net->DropoutPercent = 20;
  
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
	int max = layers+1;

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

/* back-propogate errors */
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

	/* error on the output units */
	net->BPerror = fabs(net->BPerrorTotal / net->NoOfOutputs);

	/* update the running average */
	if (net->BPerrorAverage == DEEPLEARN_UNKNOWN_ERROR) {
		net->BPerrorAverage = net->BPerror;
	}
	else {
		net->BPerrorAverage =
			(net->BPerrorAverage*0.98f) +
			(net->BPerror*0.02f);
	}	

	/* back-propogate through the hidden layers */
	for (l = net->HiddenLayers-1; l >= 0; l--) {	
		for (i = 0; i < net->NoOfHiddens; i++) {
			n = net->hiddens[l][i];
			bp_neuron_backprop(n);
			net->BPerrorTotal += n->BPerror;
		}
	}

	/* overall average error */
	net->BPerrorTotal =
		fabs(net->BPerrorTotal /
			 (net->NoOfOutputs + net->NoOfHiddens));

	/* increment the number of training itterations */
	if (net->itterations < UINT_MAX) {
		net->itterations++;
	}
}

/* adjust connection weights and bias values */
void bp_learn(bp * net)
{  
	int i,l;
	bp_neuron * n;

	/* hidden layers */
	for (l = 0; l < net->HiddenLayers; l++) {	
		for (i = 0; i < net->NoOfHiddens; i++) {
			n = net->hiddens[l][i];
			bp_neuron_learn(n,net->learningRate);
		}
	}

	/* output layer */
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

/* Set the unputs of the network from a patch within an image.
   It is assumed that the image is mono (1 byte per pixel) */
void bp_inputs_from_image_patch(bp * net,
								unsigned char * img,
								int image_width, int image_height,
								int tx, int ty)
{
	int px,py,i=0,n;
	int patch_size = (int)sqrt(net->NoOfInputs);

	assert(patch_size*patch_size <= net->NoOfInputs);

	/* set the inputs */
	for (py = ty; py < ty+patch_size; py++) {
		if (py >= image_height) break;
		for (px = tx; px < tx+patch_size; px++, i++) {
			if (px >= image_width) break;
			n = (py*image_width) + px;
			bp_set_input(net, i, 0.25f + (img[n]*0.5f/255.0f));
		}
	}
}

/* Set the inputs of the network from an image.
   It is assumed that the image is mono (1 byte per pixel) */
void bp_inputs_from_image(bp * net,
						  unsigned char * img,
						  int image_width, int image_height)
{
	int px,py,i=0;

	/* check that the number of inputs is the same as the
	   number of pixels */
	assert(net->NoOfInputs == image_width*image_height);

	/* set the inputs */
	for (py = 0; py < image_height; py++) {
		for (px = 0; px < image_width; px++, i++) {
			bp_set_input(net, i, 0.25f + (img[i]*0.5f/255.0f));
		}
	}
}

/* plots weight matrices within an image */
void bp_plot_weights(bp * net,
					 char * filename,
					 int image_width, int image_height,
					 int input_image_width)
{
	int layer, neurons_x, neurons_y, ty, by, h, x, y, ix, iy;
	int wx, wy, inputs_x, inputs_y, n, i, unit, no_of_neurons;
	int no_of_weights,wdth;
	float neuronx, neurony,dw;
	bp_neuron ** neurons, * curr_neuron;
	unsigned char * img;

	/* allocate memory for the image */
	img = (unsigned char*)malloc(image_width*image_height*3);

	/* clear the image with a white background */
	memset((void*)img,'\255',image_width*image_height*3);

	/* dimension of the neurons matrix for each layer */
	neurons_x = (int)sqrt(net->NoOfHiddens);
	neurons_y = (net->NoOfHiddens/neurons_x);

	/* dimensions of the weight matrix */
	if (input_image_width <= 0) {
		inputs_x = (int)sqrt(net->NoOfInputs);
	}
	else {
		inputs_x = input_image_width;
	}
	inputs_y = (net->NoOfInputs/inputs_x);

	no_of_weights = net->NoOfInputs;;

	/* plot the inputs */
	ty = 0;
	by = image_height/(net->HiddenLayers+3);
	h = (by-ty)*95/100;
	wdth = h;
	if (wdth>=image_width) wdth=image_width;
	for (y = 0; y < h; y++) {
		iy = y*inputs_y/h;
		for (x = 0; x < wdth; x++) {
			ix = x*inputs_x/wdth;
			unit = (iy*inputs_x) + ix;
			if (unit < net->NoOfInputs) {
				n = (y*image_width + x)*3;
				img[n] = (unsigned char)(net->inputs[unit]->value*255);
				img[n+1] = img[n];
				img[n+2] = img[n];
			}
		}
	}

	for (layer = 0; layer < net->HiddenLayers+1; layer++) {

		/* vertical top and bottom coordinates */
		ty = (layer+1)*image_height/(net->HiddenLayers+3);
		by = (layer+2)*image_height/(net->HiddenLayers+3);
		h = (by-ty)*95/100;

		/* number of patches across and down for the final layer */
		if (layer == net->HiddenLayers) {
			neurons_x = (int)sqrt(net->NoOfOutputs);
			neurons_y = (net->NoOfOutputs/neurons_x);
			neurons = net->outputs;
			no_of_neurons = net->NoOfOutputs;
		}
		else {
			neurons = net->hiddens[layer];
			no_of_neurons = net->NoOfHiddens;
		}

		/* for every pixel within the region */
		for (y = ty; y < by; y++) {
			neurony = (y-ty)*neurons_y/(float)h;
			/* y coordinate within the weights */
			wy = (neurony - (int)neurony)*inputs_y;
			for (x = 0; x < image_width; x++) {
				neuronx = x*neurons_x/(float)image_width;
				/* x coordinate within the weights */
				wx = (neuronx - (int)neuronx)*inputs_x;
				/* coordinate within the image */
				n = ((y * image_width) + x)*3;
				/* weight index */
				i = (wy*inputs_x) + wx;
				if (i < no_of_weights) {
					/* neuron index */
					unit = ((int)neurony*neurons_x) + (int)neuronx;
					if (unit < no_of_neurons)  {
						curr_neuron = neurons[unit];
						dw = curr_neuron->max_weight - 
							curr_neuron->min_weight;
						if (dw > 0.0001f) {
							img[n] =
								(int)((curr_neuron->weights[i] -
									   curr_neuron->min_weight)*255/dw);
							img[n+1] = img[n];
							img[n+2] = img[n];
						}
						else {
							img[n] =
								(int)(curr_neuron->weights[i]*255);
							img[n+1] = img[n];
							img[n+2] = img[n];
						}
					}
				}
			}
		}
		/* dimensions of the weight matrix for the next layer */
		inputs_x = (int)sqrt(net->NoOfHiddens);
		inputs_y = (net->NoOfHiddens/inputs_x);
		no_of_weights = net->NoOfHiddens;;
	}

	ty = (net->HiddenLayers+2)*image_height/(net->HiddenLayers+3);
	by = (net->HiddenLayers+3)*image_height/(net->HiddenLayers+3);
	h = (by-ty)*95/100;

	inputs_x = (int)sqrt(net->NoOfOutputs);
	inputs_y = (net->NoOfOutputs/inputs_x);

	wdth = h;
	if (wdth >= image_width) wdth = image_width;
	for (y = 0; y < h; y++) {
		iy = y*inputs_y/h;
		for (x = 0; x < wdth; x++) {
			ix = x*inputs_x/wdth;
			unit = (iy*inputs_x) + ix;
			if (unit < net->NoOfOutputs) {
				n = ((ty+y)*image_width + x)*3;
				img[n] = (unsigned char)(net->outputs[unit]->value*255);
				img[n+1] = img[n];
				img[n+2] = img[n];
			}
		}
	}

	/* write the image to file */
	deeplearn_write_png(filename,
						image_width, image_height, img);

	/* free the image memory */
	free(img);
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

/* clears the exclusion flags on neurons which have dropped out */
static void bp_clear_dropouts(bp * net)
{
	int l,i;

	if (net->DropoutPercent==0) return;

	/* clear exclusions */
	for (l = 0; l < net->HiddenLayers; l++) {
		for (i = 0; i < net->NoOfHiddens; i++) {
			net->hiddens[l][i]->excluded = 0;
		}
	}
}

/* sets exclusion flags to cause neurons to drop out */
static void bp_dropouts(bp * net)
{
	int l,i,no_of_dropouts,hidden_units,n;

	if (net->DropoutPercent==0) return;

	/* total number of hidden units */
	hidden_units = net->HiddenLayers * net->NoOfHiddens;

	/* total number of dropouts */
	no_of_dropouts = net->DropoutPercent*hidden_units/100;

	/* set the exclusion flags */
	for (n = 0; n < no_of_dropouts; n++) {
		l = rand_num(&net->random_seed)%net->HiddenLayers;
		i = rand_num(&net->random_seed)%net->NoOfHiddens;
		net->hiddens[l][i]->excluded = 1;
	}
}

void bp_update(bp * net)
{
	bp_dropouts(net);
	bp_feed_forward(net);
    bp_backprop(net);
	bp_learn(net);
	bp_clear_dropouts(net);
}

static void bp_update_autocoder(bp * net)
{
	int i;

	/* number of input and output units should be the same */
	assert(net->NoOfInputs == net->NoOfOutputs);

	/* set the target outputs to be the same as the inputs */
	for (i = 0; i < net->NoOfInputs; i++) {
		bp_set_output(net,i,net->inputs[i]->value);
	}

	/* run the autocoder */
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

	autocoder->DropoutPercent = net->DropoutPercent;
	autocoder->learningRate = net->learningRate;
}

/* pre-trains a hidden layer using an autocoder */
void bp_pretrain(bp * net, bp * autocoder, int hidden_layer)
{
	int i;
	float hidden_value;

	/* feed forward to the given hidden layer */
	if (hidden_layer > 0) {
		bp_feed_forward_layers(net, hidden_layer);
	}

	if (hidden_layer > 0) {
		/* check that the number of inputs is valid */
		assert(net->NoOfHiddens == autocoder->NoOfInputs);

		/* copy the hidden unit values to the inputs
		   of the autocoder */
		for (i = 0; i < net->NoOfHiddens; i++) {
			hidden_value = bp_get_hidden(net, hidden_layer, i);
			assert(hidden_value > 0);
			bp_set_input(autocoder,i, hidden_value);
		}
	}
	else {
		/* check that the number of inputs is valid */
		assert(autocoder->NoOfInputs == net->NoOfInputs);

		/* copy the input unit values to the inputs
		   of the autocoder */
		for (i = 0; i < net->NoOfInputs; i++) {
			bp_set_input(autocoder, i,
						 bp_get_input(net, i));
		}
	}

	/* run the autocoder */
	bp_update_autocoder(autocoder);
}

/* save a network to file */
int bp_save(FILE * fp, bp * net)
{
	int retval,i,l;

	retval = fwrite(&net->itterations, sizeof(unsigned int), 1, fp);
	retval = fwrite(&net->NoOfInputs, sizeof(int), 1, fp);
	retval = fwrite(&net->NoOfHiddens, sizeof(int), 1, fp);
	retval = fwrite(&net->NoOfOutputs, sizeof(int), 1, fp);
	retval = fwrite(&net->HiddenLayers, sizeof(int), 1, fp);
	retval = fwrite(&net->learningRate, sizeof(float), 1, fp);
	retval = fwrite(&net->noise, sizeof(float), 1, fp);
	retval = fwrite(&net->BPerrorAverage, sizeof(float), 1, fp);
	retval = fwrite(&net->DropoutPercent, sizeof(float), 1, fp);

	for (l = 0; l < net->HiddenLayers; l++) {
		for (i = 0; i < net->NoOfHiddens; i++) {
			bp_neuron_save(fp,net->hiddens[l][i]);
		}
	}
	for (i = 0; i < net->NoOfOutputs; i++) {
		bp_neuron_save(fp,net->outputs[i]);
	}

	return retval;
}

/* load a network from file */
int bp_load(FILE * fp, bp * net,
			unsigned int * random_seed)
{
	int retval,i,l;
	int no_of_inputs=0, no_of_hiddens=0, no_of_outputs=0;
	int hidden_layers=0;
	float learning_rate=0, noise=0, BPerrorAverage=0;
	float DropoutPercent=0;
	unsigned int itterations=0;

	retval = fread(&itterations, sizeof(unsigned int), 1, fp);
	retval = fread(&no_of_inputs, sizeof(int), 1, fp);
	retval = fread(&no_of_hiddens, sizeof(int), 1, fp);
	retval = fread(&no_of_outputs, sizeof(int), 1, fp);
	retval = fread(&hidden_layers, sizeof(int), 1, fp);
	retval = fread(&learning_rate, sizeof(float), 1, fp);
	retval = fread(&noise, sizeof(float), 1, fp);
	retval = fread(&BPerrorAverage, sizeof(float), 1, fp);
	retval = fread(&DropoutPercent, sizeof(float), 1, fp);

	bp_init(net, no_of_inputs, no_of_hiddens,
			hidden_layers, no_of_outputs,
			random_seed);

	for (l = 0; l < net->HiddenLayers; l++) {
		for (i = 0; i < net->NoOfHiddens; i++) {
			bp_neuron_load(fp,net->hiddens[l][i]);
		}
	}
	for (i = 0; i < net->NoOfOutputs; i++) {
		bp_neuron_load(fp,net->outputs[i]);
	}

	net->learningRate = learning_rate;
	net->noise = noise;
	net->BPerrorAverage = BPerrorAverage;
	net->BPerror = BPerrorAverage;
	net->BPerrorTotal = BPerrorAverage;
	net->itterations = itterations;
	net->DropoutPercent = DropoutPercent;

	return retval;
}

/* compares two networks and returns a greater than zero
   value if they are the same */
int bp_compare(bp * net1, bp * net2)
{
	int retval,i,l;

	if (net1->NoOfInputs != net2->NoOfInputs) {
		return -1;
	}
	if (net1->NoOfHiddens != net2->NoOfHiddens) {
		return -2;
	}
	if (net1->NoOfOutputs != net2->NoOfOutputs) {
		return -3;
	}
	if (net1->HiddenLayers != net2->HiddenLayers) {
		return -4;
	}
	if (net1->learningRate != net2->learningRate) {
		return -5;
	}
	if (net1->noise != net2->noise) {
		return -6;
	}
	for (l = 0; l < net1->HiddenLayers; l++) {
		for (i = 0; i < net1->NoOfHiddens; i++) {
			retval =
				bp_neuron_compare(net1->hiddens[l][i],
								  net2->hiddens[l][i]);
			if (retval == 0) return -7;
		}
	}
	for (i = 0; i < net1->NoOfOutputs; i++) {
		retval = bp_neuron_compare(net1->outputs[i], net2->outputs[i]);
		if (retval == 0) return -8;
	}
	if (net1->itterations != net2->itterations) {
		return -9;
	}
	if (net1->BPerrorAverage != net2->BPerrorAverage) {
		return -9;
	}
	if (net1->DropoutPercent!= net2->DropoutPercent) {
		return -10;
	}
	return 1;
}
