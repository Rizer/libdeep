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

#include "deeplearn_images.h"

void deeplearn_read_png(char * filename, png_t * ptr,
						unsigned char ** buffer)
{
    int i,j,retval;
    unsigned char * buff = NULL, * buff2 = NULL;

    png_init(0,0);
    retval = png_open_file(ptr, filename);
    if (retval != PNG_NO_ERROR)
    {
        printf("Failed to open file %s\n", filename);

        switch(retval)
        {
        case PNG_FILE_ERROR:
        {
            printf("File error\n");
            break;
        }
        case PNG_HEADER_ERROR:
        {
            printf("Header error\n");
            break;
        }
        case PNG_IO_ERROR:
        {
            printf("IO error\n");
            break;
        }
        case PNG_EOF_ERROR:
        {
            printf("EOF error\n");
            break;
        }
        case PNG_CRC_ERROR:
        {
            printf("CRC error\n");
            break;
        }
        case PNG_MEMORY_ERROR:
        {
            printf("Memory error\n");
            break;
        }
        case PNG_ZLIB_ERROR:
        {
            printf("Zlib error\n");
            break;
        }
        case PNG_UNKNOWN_FILTER:
        {
            printf("Unknown filter\n");
            break;
        }
        case PNG_NOT_SUPPORTED:
        {
            printf("PNG format not supported\n");
            break;
        }
        case PNG_WRONG_ARGUMENTS:
        {
            printf("Wrong arguments\n");
            break;
        }
        }
		*buffer = NULL;
        return;
    }

	/* single byte per pixel */
    if (ptr->bpp == 1)
    {
		buff =
			(unsigned char *)malloc(ptr->width *
									ptr->height *
									sizeof(unsigned char));
		png_get_data(ptr, buff);

        buff2 =
			(unsigned char *)malloc(ptr->width * ptr->height * 3);
		j = 0;
        for (i = 0; i < ptr->width * ptr->height; i++, j += 3) {
            buff2[j] = buff[i];
            buff2[j+1] = buff[i];
            buff2[j+2] = buff[i];
        }
        free(buff);
        *buffer = buff2;
		return;
	}

    if (ptr->bpp < 3)
    {
        printf("Not enought bytes per pixel (%d)\n", ptr->bpp);
		*buffer = NULL;
        return;
    }

    buff =
		(unsigned char *)malloc(ptr->width * ptr->height * ptr->bpp);
    png_get_data(ptr, buff);

    if (ptr->bpp > 3)
    {
        buff2 =
			(unsigned char *)malloc(ptr->width * ptr->height * 3);
        j=0;
        for (i=0; i<ptr->width * ptr->height * ptr->bpp;
			 i+=ptr->bpp,j+=3)
        {
            buff2[j] = buff[i];
            buff2[j+1] = buff[i+1];
            buff2[j+2] = buff[i+2];
        }
        free(buff);
        *buffer = buff2;
    }
	else {
		*buffer = buff;
	}

    return;
}

int deeplearn_write_png(char* filename,
						int width, int height,
						unsigned char *buffer)
{
    png_t png;
    FILE * fp = fopen(filename, "wb");
    if (fp == NULL)
    {
        fprintf(stderr,
				"Could not open file %s for writing\n", filename);
        return 1;
    }
    fclose(fp);

    png_init(0,0);
    png_open_file_write(&png, filename);
    png_set_data(&png, width, height, 8, PNG_TRUECOLOR, buffer);
    png_close_file(&png);

    return 0;
}

static int number_of_images(char * images_directory,
							char * extension)
{
	int ctr,no_of_images = 0;
    struct dirent **namelist;
    int n,len;

	/* get image filenames */
	n = scandir(images_directory, &namelist, 0, alphasort);
    if (n >= 0) {
		/* count the number of image files */
		ctr = n;
        while (ctr--) {
			/* is the filename long enough? */
			len = strlen(namelist[ctr]->d_name);
			if (len > 4) {
				/* is this a png image? */
				if ((namelist[ctr]->d_name[len-4]=='.') &&
					(namelist[ctr]->d_name[len-3]==extension[0]) &&
					(namelist[ctr]->d_name[len-2]==extension[1]) &&
					(namelist[ctr]->d_name[len-1]==extension[2])) {
					no_of_images++;
				}
			}
            free(namelist[ctr]);
		}
        free(namelist);
	}
	return no_of_images;
}

/* downsample a colour image to a mono fixes size image */
static void deeplearn_downsample(unsigned char * img,
								 int width, int height,
								 unsigned char * downsampled,
								 int downsampled_width,
								 int downsampled_height)
{
	int x,y,n2,xx,yy,n=0;

	for (y = 0; y < downsampled_height; y++) {
		yy = y * height / downsampled_height;
		for (x = 0; x < downsampled_width; x++, n++) {
			xx = x * width / downsampled_width;
			n2 = ((yy*width) + xx)*3;
			downsampled[n] = (img[n2]+img[n2+1]+img[n2+2])/3;
		}
	}
}

int deeplearn_load_training_images(char * images_directory,
								   unsigned char *** images,
								   int width, int height)
{
	int ctr,no_of_images = 0;
    struct dirent **namelist;
    int n,len;
	unsigned char * img, * downsampled;
	png_t ptr;
	char * extension = "png";
	char filename[256];

	memset((void*)&ptr,'\0',sizeof(&ptr));

	/* how many images are there? */
	no_of_images = number_of_images(images_directory, extension);
	if (no_of_images == 0) {
		return 0;
	}

	/* allocate an array for the images */
	*images =
		(unsigned char**)malloc(no_of_images*
								sizeof(unsigned char*));

	/* get image filenames */
	no_of_images = 0;
	n = scandir(images_directory, &namelist, 0, alphasort);
    if (n >= 0) {
		/* for every filename */
		ctr = n;
        while (ctr--) {
			/* is the filename long enough? */
			len = strlen(namelist[ctr]->d_name);
			if (len > 4) {
				sprintf(filename,"%s/%s",
						images_directory,namelist[ctr]->d_name);
				len = strlen(filename);
				/* is this a png image? */
				if ((filename[len-4]=='.') &&
					(filename[len-3]==extension[0]) &&
					(filename[len-2]==extension[1]) &&
					(filename[len-1]==extension[2])) {

					/* obtain an image from the filename */
					deeplearn_read_png(filename, &ptr, &img);

					/* was an image returned? */
					if (img != NULL) {
						/* create a fixed size image */
						downsampled =
							(unsigned char*)malloc(width*height*
												   sizeof(unsigned char));
						deeplearn_downsample(img, ptr.width, ptr.height,
											 downsampled, width, height);

						*images[no_of_images] = downsampled;
						printf("Assigned %d\n",no_of_images);

						/* free the original image */
						free(img);
					}
					else {
						*images[no_of_images] = NULL;
					}
					png_close_file(&ptr);
					no_of_images++;
				}
			}
            free(namelist[ctr]);
        }
        free(namelist);
    }

	return no_of_images;
}
