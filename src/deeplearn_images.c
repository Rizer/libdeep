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

unsigned char * deeplearn_read_png(char * filename, png_t * ptr)
{
    int i,j,retval;
    unsigned char * buffer = NULL, * buffer2 = NULL;

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
        return buffer;
    }

    if (ptr->bpp < 3)
    {
        printf("Not enought bytes per pixel\n");
        return buffer;
    }

    buffer =
		(unsigned char *)malloc(ptr->width * ptr->height * ptr->bpp);
    png_get_data(ptr, buffer);

    if (ptr->bpp > 3)
    {
        buffer2 =
			(unsigned char *)malloc(ptr->width * ptr->height * 3);
        j=0;
        for (i=0; i<ptr->width * ptr->height * ptr->bpp;
			 i+=ptr->bpp,j+=3)
        {
            buffer2[j] = buffer[i];
            buffer2[j+1] = buffer[i+1];
            buffer2[j+2] = buffer[i+2];
        }
        free(buffer);
        buffer = buffer2;
    }

    return buffer;
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
