/*
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

#include "tests_backprop.h"

static void test_backprop_neuron_init()
{
	bp_neuron n;
	int no_of_inputs=10;
	unsigned int random_seed = 123;

	printf("test_backprop_neuron_init...");

	bp_neuron_init(&n, no_of_inputs, &random_seed);
	bp_neuron_free(&n);

	printf("Ok\n");
}

static void test_backprop_inputs_from_image()
{
	bp net;
	int image_width=10;
	int image_height=10;
	int x,y,i=0;
	unsigned char * img;
	int no_of_inputs=image_width*image_height;
	int no_of_hiddens=4*4;
	int hidden_layers=2;
	int no_of_outputs=2*2;
	unsigned int random_seed = 123;

	printf("test_backprop_inputs_from_image...");

	/* create a mono image */
	img = (unsigned char*)malloc(image_width*image_height);
	for (y = 0; y < image_height; y++) {
		for (x = 0; x < image_width; x++,i++) {
			img[i] = i%256;
		}
	}

	/* create a network */
	bp_init(&net,
			no_of_inputs, no_of_hiddens,
			hidden_layers,
			no_of_outputs, &random_seed);
	assert((&net)->inputs!=0);
	assert((&net)->hiddens!=0);
	assert((&net)->outputs!=0);

	/* set inputs to zero */
	for (i = 0; i < no_of_inputs; i++) {
		(&net)->inputs[i]->value = 0;
	}

	/* insert the image into the input units */
	bp_inputs_from_image(&net, img, image_width, image_height);

	/* check that the imputs are within range */
	for (i = 0; i < no_of_inputs; i++) {
		assert((&net)->inputs[i]->value > 0.1f);
		assert((&net)->inputs[i]->value < 0.9f);
	}

	/* free the memory */
	free(img);
	bp_free(&net);

	printf("Ok\n");
}

static void test_backprop_neuron_copy()
{
	bp_neuron n1, n2;
	int retval, no_of_inputs=10;
	unsigned int random_seed = 123;

	printf("test_backprop_neuron_copy...");

	bp_neuron_init(&n1, no_of_inputs, &random_seed);
	bp_neuron_init(&n2, no_of_inputs, &random_seed);

	bp_neuron_copy(&n1, &n2);

	retval = bp_neuron_compare(&n1, &n2);
	if (retval != 1) {
		printf("\nretval %d\n", retval);
	}
	assert(retval == 1);

	bp_neuron_free(&n1);
	bp_neuron_free(&n2);

	printf("Ok\n");
}

static void test_backprop_init()
{
	bp net;
	int no_of_inputs=10;
	int no_of_hiddens=4;
	int hidden_layers=2;
	int no_of_outputs=2;
	unsigned int random_seed = 123;

	printf("test_backprop_init...");

	bp_init(&net,
			no_of_inputs, no_of_hiddens,
			hidden_layers,
			no_of_outputs, &random_seed);
	assert((&net)->inputs!=0);
	assert((&net)->hiddens!=0);
	assert((&net)->outputs!=0);
	bp_free(&net);

	printf("Ok\n");
}

static void test_backprop_feed_forward()
{
	bp net;
	int no_of_inputs=10;
	int no_of_hiddens=4;
	int hidden_layers=2;
	int no_of_outputs=5;
	int i;
	unsigned int random_seed = 123;

	printf("test_backprop_feed_forward...");

	bp_init(&net,
			no_of_inputs, no_of_hiddens,
			hidden_layers,
			no_of_outputs, &random_seed);
	assert((&net)->inputs!=0);
	assert((&net)->hiddens!=0);
	assert((&net)->outputs!=0);

	/* set some inputs */
	for (i = 0; i < no_of_inputs; i++) {
		bp_set_input(&net, i, i/(float)no_of_inputs);
	}
	/* clear the outputs */
	for (i = 0; i < no_of_outputs; i++) {
		(&net)->outputs[i]->value = 999;
	}

	/* feed forward */
	bp_feed_forward(&net);

	/* check for non-zero outputs */
	for (i = 0; i < no_of_outputs; i++) {
		assert((&net)->outputs[i]->value != 999);
	}

	bp_free(&net);

	printf("Ok\n");
}

static void test_backprop()
{
	bp net;
	int no_of_inputs=10;
	int no_of_hiddens=4;
	int hidden_layers=2;
	int no_of_outputs=5;
	int i,l;
	unsigned int random_seed = 123;

	printf("test_backprop...");

	bp_init(&net,
			no_of_inputs, no_of_hiddens,
			hidden_layers,
			no_of_outputs, &random_seed);
	assert((&net)->inputs!=0);
	assert((&net)->hiddens!=0);
	assert((&net)->outputs!=0);

	/* set some inputs */
	for (i = 0; i < no_of_inputs; i++) {
		bp_set_input(&net, i, i/(float)no_of_inputs);
		(&net)->inputs[i]->BPerror = 999;
	}
	for (l = 0; l < hidden_layers; l++) {
		for (i = 0; i < no_of_hiddens; i++) {
			(&net)->hiddens[l][i]->BPerror = 999;
		}
	}
	/* set some target outputs */
	for (i = 0; i < no_of_outputs; i++) {
		(&net)->outputs[i]->BPerror = 999;
		bp_set_output(&net, i, i/(float)no_of_inputs);
	}

	/* feed forward */
	bp_feed_forward(&net);
	bp_backprop(&net);

	/* check for non-zero backprop error values */
	for (i = 0; i < no_of_inputs; i++) {
		assert((&net)->inputs[i]->BPerror != 999);
	}
	for (l = 0; l < hidden_layers; l++) {
		for (i = 0; i < no_of_hiddens; i++) {
			assert((&net)->hiddens[l][i]->BPerror != 999);
		}
	}
	for (i = 0; i < no_of_outputs; i++) {
		assert((&net)->outputs[i]->BPerror != 999);
	}

	bp_free(&net);

	printf("Ok\n");
}

static void test_backprop_update()
{
	bp net;
	int no_of_inputs=10;
	int no_of_hiddens=4;
	int hidden_layers=2;
	int no_of_outputs=5;
	int i;
	unsigned int random_seed = 123;

	printf("test_backprop_update...");

	bp_init(&net,
			no_of_inputs, no_of_hiddens,
			hidden_layers,
			no_of_outputs, &random_seed);
	assert((&net)->inputs!=0);
	assert((&net)->hiddens!=0);
	assert((&net)->outputs!=0);

	/* set some inputs */
	for (i = 0; i < no_of_inputs; i++) {
		bp_set_input(&net, i, i/(float)no_of_inputs);
	}

	for (i = 0; i < 100; i++) {
		bp_update(&net);
	}

	bp_free(&net);

	printf("Ok\n");
}

static void test_backprop_training()
{
	bp * net;
	int no_of_inputs=2;
	int no_of_hiddens=2;
	int hidden_layers=1;
	int no_of_outputs=1;
	int itt,example;
	unsigned int random_seed = 123;
	float state_TRUE = 0.8f;
	float state_FALSE = 0.2f;

	printf("test_backprop_training...");

	net = (bp*)malloc(sizeof(bp));
	bp_init(net,
			no_of_inputs, no_of_hiddens,
			hidden_layers,
			no_of_outputs, &random_seed);
	assert(net->inputs!=0);
	assert(net->hiddens!=0);
	assert(net->outputs!=0);

	/* training */
	example=0;
	for (itt = 0; itt < 500000; itt++, example++) {
		if (example>=4) example=0;

		/* select an example from the XOR truth table */
		switch(example) {
		case 0: {
			bp_set_input(net, 0, state_FALSE);
			bp_set_input(net, 1, state_FALSE);
			bp_set_output(net, 0, state_FALSE);
			break;
		}
		case 1: {
			bp_set_input(net, 0, state_TRUE);
			bp_set_input(net, 1, state_FALSE);
			bp_set_output(net, 0, state_TRUE);
			break;
		}
		case 2: {
			bp_set_input(net, 0, state_FALSE);
			bp_set_input(net, 1, state_TRUE);
			bp_set_output(net, 0, state_TRUE);
			break;
		}
		case 3: {
			bp_set_input(net, 0, state_TRUE);
			bp_set_input(net, 1, state_TRUE);
			bp_set_output(net, 0, state_FALSE);
			break;
		}
		}

		/* train on the example */
		bp_update(net);
	}

	bp_set_input(net, 0, state_FALSE);
	bp_set_input(net, 1, state_FALSE);
	bp_feed_forward(net);
	if (bp_get_output(net, 0) >= 0.5f) {
		printf("\n%.5f\n",bp_get_output(net, 0));
	}
	assert(bp_get_output(net, 0) < 0.5f);

	bp_set_input(net, 0, state_FALSE);
	bp_set_input(net, 1, state_TRUE);
	bp_feed_forward(net);
	if (bp_get_output(net, 0) <= 0.5f) {
		printf("\n%.5f\n",bp_get_output(net, 0));
	}
	assert(bp_get_output(net, 0) > 0.5f);

	bp_set_input(net, 0, state_TRUE);
	bp_set_input(net, 1, state_FALSE);
	bp_feed_forward(net);
	if (bp_get_output(net, 0) <= 0.5f) {
		printf("\n%.5f\n",bp_get_output(net, 0));
	}
	assert(bp_get_output(net, 0) > 0.5f);

	bp_set_input(net, 0, state_FALSE);
	bp_set_input(net, 1, state_FALSE);
	bp_feed_forward(net);
	if (bp_get_output(net, 0) >= 0.5f) {
		printf("\n%.5f\n",bp_get_output(net, 0));
	}
	assert(bp_get_output(net, 0) < 0.5f);

	bp_free(net);
	free(net);

	printf("Ok\n");
}

static void test_backprop_autocoder()
{
	bp autocoder;
	int itt,i,j;
	int no_of_inputs=10;
	int no_of_hiddens=4;
	int no_of_outputs=10;
	unsigned int random_seed = 123;
	float tot;

	printf("test_backprop_autocoder...");

	/* create the autocoder */
	bp_init(&autocoder,
			no_of_inputs,
			no_of_hiddens,1,
			no_of_outputs,
			&random_seed);

	autocoder.learningRate = 0.5f;

	/* run the autocoder for some itterations */
	for (itt = 0; itt < 100; itt++) {
		/* set the inputs */
		for (i = 0; i < no_of_inputs; i++) {
			bp_set_input(&autocoder,i,0.25f + (i*0.5f/(float)no_of_inputs));
			bp_set_output(&autocoder,i,0.75f - (i*0.5f/(float)no_of_inputs));
		}
		/* update */
		bp_update(&autocoder);
	}

	for (i = 0; i < no_of_hiddens; i++) {
		/* check that some errors have been back-propogated */
		assert((&autocoder)->hiddens[0][i]->BPerror != 0);
		/* check that weights have changed */
		tot = 0;
		for (j = 0; j < no_of_inputs; j++) {
			assert((&autocoder)->hiddens[0][i]->lastWeightChange[j]!=0);
			tot += fabs((&autocoder)->hiddens[0][i]->lastWeightChange[j]);
		}
		/* total weight change */
		assert(tot > 0.00001f);
	}

	bp_free(&autocoder);

	printf("Ok\n");
}


static void test_backprop_deep()
{
	bp net;
	bp autocoder;
	int l,itt,i;
	int no_of_inputs=10;
	int no_of_hiddens=4;
	int hidden_layers=3;
	int no_of_outputs=2;
	unsigned int random_seed = 123;

	printf("test_backprop_deep...");

	bp_init(&net,
			no_of_inputs, no_of_hiddens,
			hidden_layers,
			no_of_outputs, &random_seed);
	assert((&net)->inputs!=0);
	assert((&net)->hiddens!=0);
	assert((&net)->outputs!=0);

	for (l = 0; l < hidden_layers; l++) {
		/* create an autocoder for this layer */
		bp_create_autocoder(&net,l,&autocoder);
		/* do some training */
		for (itt = 0; itt < 100; itt++) {
			/* set the inputs */
			for (i = 0; i < no_of_inputs; i++) {
				bp_set_input(&net,i,i/(float)no_of_inputs);
			}
			/* update */
			bp_pretrain(&net,&autocoder,l);
		}
		/* move the autocoder hidden weights into the main network */
		bp_update_from_autocoder(&net,&autocoder,l);
		/* delete the autocoder */
		bp_free(&autocoder);
	}

	bp_free(&net);

	printf("Ok\n");
}

static void test_backprop_neuron_save_load()
{
	bp_neuron n1, n2;
	int no_of_inputs=10;
	unsigned int random_seed = 123;
	char filename[256];
	FILE * fp;

	printf("test_backprop_neuron_save_load...");

	/* create neurons */
	bp_neuron_init(&n1, no_of_inputs, &random_seed);
	bp_neuron_init(&n2, no_of_inputs, &random_seed);

	sprintf(filename,"%stemp_deep.dat",DEEPLEARN_TEMP_DIRECTORY);

	/* save the first neuron */
	fp = fopen(filename,"wb");
	assert(fp!=0);
	bp_neuron_save(fp, &n1);
	fclose(fp);

	/* load into the second neuron */
	fp = fopen(filename,"rb");
	assert(fp!=0);
	bp_neuron_load(fp, &n2);
	fclose(fp);

	/* compare the two */
	assert(bp_neuron_compare(&n1, &n2)==1);

	/* free memory */
	bp_neuron_free(&n1);
	bp_neuron_free(&n2);

	printf("Ok\n");
}

static void test_backprop_save_load()
{
	bp net1, net2;
	int no_of_inputs=10;
	int no_of_hiddens=4;
	int no_of_outputs=3;
	int hidden_layers=3;
	int retval;
	unsigned int random_seed = 123;
	char filename[256];
	FILE * fp;

	printf("test_backprop_save_load...");

	/* create network */
	bp_init(&net1,
			no_of_inputs, no_of_hiddens,
			hidden_layers, no_of_outputs,
			&random_seed);

	sprintf(filename,"%stemp_deep.dat",DEEPLEARN_TEMP_DIRECTORY);

	/* save the first network */
	fp = fopen(filename,"wb");
	assert(fp!=0);
	bp_save(fp, &net1);
	fclose(fp);

	/* load into the second network */
	fp = fopen(filename,"rb");
	assert(fp!=0);
	bp_load(fp, &net2, &random_seed);
	fclose(fp);

	/* compare the two */
	retval = bp_compare(&net1, &net2);
	if (retval<1) {
		printf("\nretval = %d\n",retval);
	}
	assert(retval==1);

	/* free memory */
	bp_free(&net1);
	bp_free(&net2);

	printf("Ok\n");
}

static void test_backprop_classification_from_filename()
{
	char classification[256];

	printf("test_backprop_classification_from_filename...");

	bp_get_classification_from_filename("class.number.png",
										classification);
	assert(strcmp(classification,"class")==0);
	bp_get_classification_from_filename("/my/directory/test.number.png",
										classification);
	assert(strcmp(classification,"test")==0);
	printf("Ok\n");
}

int run_tests_backprop()
{
	printf("\nRunning backprop tests\n");

	test_backprop_neuron_init();	
	test_backprop_neuron_copy();
	test_backprop_init();
	test_backprop_feed_forward();
	test_backprop();
	test_backprop_update();
	test_backprop_training();
	test_backprop_deep();
	test_backprop_neuron_save_load();
	test_backprop_save_load();
	test_backprop_inputs_from_image();
	test_backprop_autocoder();
	test_backprop_classification_from_filename();

	printf("All backprop tests completed\n");
	return 1;
}
