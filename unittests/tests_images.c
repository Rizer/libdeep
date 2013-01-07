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

#include "tests_images.h"

static void save_image(char * filename)
{
	int width = 80;
	int height = 80;
	unsigned char * buffer;
	int i;
	FILE * fp;

	/* allocate memory */
	buffer = (unsigned char*)malloc(width*height*3*
									sizeof(unsigned char));
	assert(buffer != 0);

	/* create a random image */
	for (i = 0; i < width*height*3; i+=3) {
		buffer[i] = i%256;
		buffer[i+1] = 255 - buffer[i];
		buffer[i+2] = buffer[i];
	}

	/* write to file */
	deeplearn_write_png(filename, width, height, buffer);

	/* free memory */
	free(buffer);

	/* check that the file has saved */
	fp = fopen(filename,"rb");
	assert(fp);
	fclose(fp);
}

static void test_save_image()
{
	char filename[256], commandstr[256];

	printf("test_save_image...");

	sprintf(filename,"%stemp_img.png",DEEPLEARN_TEMP_DIRECTORY);
	save_image(filename);

	/* remove the image */
	sprintf(commandstr,"rm -f %stemp_img.png",
			DEEPLEARN_TEMP_DIRECTORY);
	system(commandstr);

	printf("Ok\n");
}

static void test_load_image()
{
	char filename[256], commandstr[256];
	unsigned char * buffer;
	png_t ptr;
	int i;

	printf("test_load_image...");

	/* save a tests image */
	sprintf(filename,"%stemp_deeplearn_img.png",DEEPLEARN_TEMP_DIRECTORY);
	save_image(filename);

	/* load image from file */
	deeplearn_read_png(filename, &ptr, &buffer);

	/* check image properties */
	assert(ptr.width==80);
	assert(ptr.height==80);
	assert(ptr.bpp==3);

	/* check the pixels */
	for (i = 0; i < ptr.width*ptr.height*3; i+=3) {
		assert(buffer[i] == i%256);
		assert(buffer[i+1] == 255 - buffer[i]);
		assert(buffer[i+2] == buffer[i]);
	}

	/* free memory */
	free(buffer);

	/* remove the image */
	sprintf(commandstr,"rm -f %stemp_deeplearn_img.png",
			DEEPLEARN_TEMP_DIRECTORY);
	system(commandstr);

	printf("Ok\n");
}

static void test_load_training_images()
{
	char filename[256];
	unsigned char ** images=NULL;
	int im;
	int no_of_images = 3;
	int no_of_images2;
	int width=40,height=40;
	char commandstr[256],str[256];

	printf("test_load_training_images...");

	/* create a directory for the images */
	sprintf(commandstr,"mkdir %sdeeplearn_test_images",
			DEEPLEARN_TEMP_DIRECTORY);
	system(commandstr);

	/* save a tests images */
	for (im = 0; im < no_of_images; im++) {
		sprintf(filename,"%sdeeplearn_test_images/img%d.png",
				DEEPLEARN_TEMP_DIRECTORY, im);
		save_image(filename);
	}

	sprintf(str,"%sdeeplearn_test_images",
			DEEPLEARN_TEMP_DIRECTORY);

	/* load the images */
	no_of_images2 =
		deeplearn_load_training_images(str,
									   &images, width, height);
	assert(no_of_images == no_of_images2);
	assert(images!=NULL);
	
	/* free memory */
	for (im = 0; im < no_of_images; im++) {
		free(images[im]);
	}
	free(images);

	/* remove the images */
	sprintf(commandstr,"rm -rf %sdeeplearn_test_images",
			DEEPLEARN_TEMP_DIRECTORY);
	system(commandstr);

	printf("Ok\n");
}

int run_tests_images()
{
	printf("\nRunning images tests\n");

	test_save_image();
	test_load_image();
	test_load_training_images();

	printf("All images tests completed\n");
	return 1;
}
