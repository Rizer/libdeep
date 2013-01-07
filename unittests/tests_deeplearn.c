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

#include "tests_deeplearn.h"

static void test_deeplearn_init()
{
	deeplearn learner;
	int no_of_inputs=10;
	int no_of_hiddens=4;
	int hidden_layers=2;
	int no_of_outputs=2;
	unsigned int random_seed = 123;

	printf("test_deeplearn_init...");

	/* create the learner */
	deeplearn_init(&learner,
				   no_of_inputs, no_of_hiddens,
				   hidden_layers,
				   no_of_outputs, &random_seed);

	assert((&learner)->net!=0);
	assert((&learner)->autocoder!=0);

	/* free memory */
	deeplearn_free(&learner);

	printf("Ok\n");
}

static void test_deeplearn_update()
{
	deeplearn learner, learner2;
	int no_of_inputs=10;
	int no_of_hiddens=4;
	int hidden_layers=2;
	int no_of_outputs=2;
	int itt,i,retval;
	unsigned int random_seed = 123;
	float max_backprop_error = 0.03f;
	int itterations[3];
	char filename[256];
	FILE * fp;

	printf("test_deeplearn_update...");

	itterations[0] = 0;
	itterations[1] = 0;
	itterations[2] = 0;

	/* create the learner */
	deeplearn_init(&learner,
				   no_of_inputs, no_of_hiddens,
				   hidden_layers,
				   no_of_outputs, &random_seed);

	assert((&learner)->net!=0);
	assert((&learner)->autocoder!=0);

	/* perform pre-training with an autocoder */
	for (itt = 0; itt < 10000; itt++) {
		for (i = 0; i < no_of_inputs; i++) {
			deeplearn_set_input(&learner,i,i/(float)no_of_inputs);
		}
		deeplearn_update(&learner, max_backprop_error);

		itterations[learner.current_hidden_layer]++;

		if (learner.current_hidden_layer==hidden_layers) {
			break;
		}
	}

	/* we expect that there will be some non-zero error */
	assert(learner.BPerror!=0);

	/* test that it took some itterations to train */
	assert(itterations[0] > 4);
	assert(itterations[1] > 4);

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

		itterations[learner.current_hidden_layer]++;

		if (learner.BPerror < max_backprop_error) {
			break;
		}
	}

	/* test that it took some itterations to
	   do the final training */
	assert(itterations[2] > 4);

	/* we expect that there will be some non-zero error */
	assert(learner.BPerror!=0);

	sprintf(filename,"%stemp_deep.dat",DEEPLEARN_TEMP_DIRECTORY);

	/* save the first learner */
	fp = fopen(filename,"wb");
	assert(fp!=0);
	deeplearn_save(fp, &learner);
	fclose(fp);

	/* load into the second learner */
	fp = fopen(filename,"rb");
	assert(fp!=0);
	deeplearn_load(fp, &learner2, &random_seed);
	fclose(fp);

	/* compare the two */
	retval = deeplearn_compare(&learner, &learner2);
	if (retval<1) {
		printf("\nretval = %d\n",retval);
	}
	assert(retval==1);

	/* save a graph */
	sprintf(filename,"%stemp_graph.png",DEEPLEARN_TEMP_DIRECTORY);
	deeplearn_plot_history(&learner,
						   filename, "Training Error",
						   1024, 480);

	/* free memory */
	deeplearn_free(&learner);
	deeplearn_free(&learner2);

	printf("Ok\n");
}

static void test_deeplearn_save_load()
{
	deeplearn learner1, learner2;
	int no_of_inputs=10;
	int no_of_hiddens=4;
	int no_of_outputs=3;
	int hidden_layers=3;
	int retval;
	unsigned int random_seed = 123;
	char filename[256];
	FILE * fp;

	printf("test_deeplearn_save_load...");

	/* create network */
	deeplearn_init(&learner1,
				   no_of_inputs, no_of_hiddens,
				   hidden_layers, no_of_outputs,
				   &random_seed);

	sprintf(filename,"%stemp_deep.dat",DEEPLEARN_TEMP_DIRECTORY);

	/* save the first learner */
	fp = fopen(filename,"wb");
	assert(fp!=0);
	deeplearn_save(fp, &learner1);
	fclose(fp);

	/* load into the second learner */
	fp = fopen(filename,"rb");
	assert(fp!=0);
	deeplearn_load(fp, &learner2, &random_seed);
	fclose(fp);

	/* compare the two */
	retval = deeplearn_compare(&learner1, &learner2);
	if (retval<1) {
		printf("\nretval = %d\n",retval);
	}
	assert(retval==1);

	/* free memory */
	deeplearn_free(&learner1);
	deeplearn_free(&learner2);

	printf("Ok\n");
}

int run_tests_deeplearn()
{
	printf("\nRunning deeplearn tests\n");

	test_deeplearn_init();
	test_deeplearn_save_load();
	test_deeplearn_update();

	printf("All deeplearn tests completed\n");
	return 1;
}
