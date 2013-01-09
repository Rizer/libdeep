
all:
	gcc -Wall -ansi -pedantic -o facerec facerec.c -ldeep -lm -lz -lpng -fopenmp

debug:
	gcc -Wall -ansi -pedantic -g -o facerec facerec.c -ldeep -lm -lz -lpng -fopenmp

clean:
	rm -f *.o facerec
