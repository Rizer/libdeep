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
#include "libdeep/deeplearn_images.h"

/* the dimensions of each face image */
int image_width = 80;
int image_height = 80;

/* the number of face images */
int no_of_images;

/* array storing the face images */
unsigned char **images;

deeplearn learner;

/* train the deep learner */
static void facerec_training()
{
	int patch_size = image_width/10;
	int no_of_inputs = patch_size*patch_size;
	int no_of_hiddens = patch_size;
	int hidden_layers=4;
	int no_of_outputs=5;
	int itt,i,tx,ty;
	unsigned int random_seed = 123;
	float max_backprop_error = 0.01f;
	char filename[256];
	char title[256];

	sprintf(title, "%s", "Face Image Training");

	/* create the learner */
	deeplearn_init(&learner,
				   no_of_inputs, no_of_hiddens,
				   hidden_layers,
				   no_of_outputs, &random_seed);

	/* perform pre-training with an autocoder */
	itt = 0;
	while (learner.current_hidden_layer < hidden_layers) {
		/* pick a random patch location */
		tx = rand_num(&random_seed)%(image_width-patch_size);
		ty = rand_num(&random_seed)%(image_height-patch_size);

		/* load the patch into the network inputs */
		deeplearn_inputs_from_image_patch(&learner,
										  images[rand_num(&random_seed)%no_of_images],
										  image_width, image_height,
										  tx, ty);

		deeplearn_update(&learner, max_backprop_error);
		itt++;
		printf("%d: %.5f\n",
			   learner.current_hidden_layer, learner.BPerror);
		if ((itt % 1000 == 0) && (itt>0)) {
			/* save a graph */
			sprintf(filename,"%s","training_error.png");
			deeplearn_plot_history(&learner,
								   filename, title,
								   1024, 480);			
		}
	}

	/* perform the final training between the last
	   hidden layer and the outputs */
	while (learner.BPerror > max_backprop_error) {
		/* pick a random patch location */
		tx = rand_num(&random_seed)%(image_width-patch_size);
		ty = rand_num(&random_seed)%(image_height-patch_size);

		/* load the patch into the network inputs */
		deeplearn_inputs_from_image_patch(&learner,
										  images[rand_num(&random_seed)%no_of_images],
										  image_width, image_height,
										  tx, ty);

		for (i = 0; i < no_of_outputs; i++) {
			deeplearn_set_output(&learner,i,
								 1.0f - (i/(float)no_of_inputs));
		}
		deeplearn_update(&learner, max_backprop_error);

		itt++;
		printf("Final: %.5f\n",learner.BPerror);
		if ((itt % 1000 == 0) && (itt>0)) {
			/* save a graph */
			sprintf(filename,"%s","training_error.png");
			deeplearn_plot_history(&learner,
								   filename, title,
								   1024, 480);			
		}
	}

	/* save a graph */
	sprintf(filename,"%s","training_error.png");
	deeplearn_plot_history(&learner,
						   filename, title,
						   1024, 480);
}

/* deallocate images */
static void free_mem(unsigned char ** images,
					 int no_of_images)
{
	int i;

	if (images==NULL) return;

	for (i = 0; i < no_of_images; i++) {
		if (images[i] != NULL) {
			free(images[i]);
			images[i] = 0;
		}
	}
	free(images);

	deeplearn_free(&learner);
}

int main(int argc, char* argv[])
{
	images = NULL;

	/* load training images into an array */
	no_of_images =
		deeplearn_load_training_images("images", &images,
									   image_width, image_height);
	
	printf("No of images: %d\n", no_of_images);

	facerec_training();

	free_mem(images, no_of_images);
	return 1;
}

