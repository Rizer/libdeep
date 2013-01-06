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

#include <stdio.h>
#include "libdeep/backprop.h"

static void test()
{
	bp net;
	bp * autocoder;
	int l,itt,i;
	int no_of_inputs=10;
	int no_of_hiddens=4;
	int hidden_layers=3;
	int no_of_outputs=2;
	unsigned int random_seed = 123;

	bp_init(&net,
			no_of_inputs, no_of_hiddens,
			hidden_layers,
			no_of_outputs, &random_seed);
	assert((&net)->inputs!=0);
	assert((&net)->hiddens!=0);
	assert((&net)->outputs!=0);

	for (l = 0; l < hidden_layers; l++) {
		/* create an autocoder for this layer */
		autocoder = bp_create_autocoder(&net,l);
		/* do some training */
		for (itt = 0; itt < 100; itt++) {
			/* set the inputs */
			for (i = 0; i < no_of_inputs; i++) {
				bp_set_input(&net,i,i/(float)no_of_inputs);
			}
			/* update */
			bp_pretrain(&net,autocoder,l);
		}
		/* move the autocoder hidden weights into the main network */
		bp_update_from_autocoder(&net,autocoder,l);
		/* delete the autocoder */
		bp_free(autocoder);
	}

	bp_free(&net);
}

int main(int argc, char* argv[])
{	
	test();
	return 1;
}

