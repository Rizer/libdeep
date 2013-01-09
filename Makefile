APP=libdeep
VERSION=1.00
RELEASE=1
SONAME=$(APP).so.0
LIBNAME=$(APP)-$(VERSION).so.0.0.$(RELEASE)
USRBASE=/usr

all:
	gcc -c -std=c99 -pedantic -fPIC -o $(APP)_neuron.o src/backprop_neuron.c -Isrc -lm -fopenmp
	gcc -c -std=c99 -pedantic -fPIC -o $(APP).o src/backprop.c -Isrc -lm -fopenmp
	gcc -c -std=c99 -pedantic -fPIC -o $(APP)learn.o src/deeplearn.c -Isrc -lm -fopenmp
	gcc -c -std=c99 -pedantic -fPIC -o $(APP)random.o src/deeplearn_random.c -Isrc -lm
	gcc -c -std=c99 -pedantic -fPIC -o $(APP)png.o src/pnglite.c -Isrc -lm -lz
	gcc -c -std=c99 -pedantic -fPIC -o $(APP)images.o src/deeplearn_images.c -Isrc -lm -lz -lpng
	gcc -shared -Wl,-soname,$(SONAME) -o $(LIBNAME) $(APP).o $(APP)_neuron.o $(APP)learn.o $(APP)random.o $(APP)png.o $(APP)images.o
#	objdump -p ${LIBNAME} | sed -n -e's/^[[:space:]]*SONAME[[:space:]]*//p' | sed -e's/\([0-9]\)\.so\./\1-/; s/\.so\.//'

debug:
	gcc -c -std=c99 -pedantic -fPIC -g -o $(APP).o src/backprop.c -Isrc -lm -fopenmp
	gcc -c -std=c99 -pedantic -fPIC -g -o $(APP)_neuron.o src/backprop_neuron.c -Isrc -lm -fopenmp
	gcc -c -std=c99 -pedantic -fPIC -g -o $(APP)learn.o src/deeplearn.c -Isrc -lm -fopenmp
	gcc -c -std=c99 -pedantic -fPIC -g -o $(APP)random.o src/deeplearn_random.c -Isrc -lm
	gcc -c -std=c99 -pedantic -fPIC -g -o $(APP)png.o src/pnglite.c -Isrc -lm -lz
	gcc -c -std=c99 -pedantic -fPIC -g -o $(APP)images.o src/deeplearn_images.c -Isrc -lm -lz -lpng
	gcc -shared -Wl,-soname,$(SONAME) -o $(LIBNAME) $(APP).o $(APP)_neuron.o $(APP)learn.o $(APP)random.o $(APP)png.o $(APP)images.o

tests:
	gcc -Wall -std=c99 -pedantic -g -o $(APP)_tests unittests/*.c src/*.c -Isrc -Iunittests -lm -lz -lpng -fopenmp

ltest:
	gcc -Wall -std=c99 -pedantic -g -o $(APP) libtest/*.c -ldeep -lm -lz -lpng -fopenmp

source:
	tar -cvzf ../$(APP)_$(VERSION).orig.tar.gz ../$(APP)-$(VERSION) --exclude=.bzr

install:
	mkdir -m 755 -p $(USRBASE)/lib
	cp $(LIBNAME) $(USRBASE)/lib
	cp man/$(APP).1.gz $(USRBASE)/share/man/man1
	mkdir -m 755 -p $(USRBASE)/include/$(APP)
	cp src/*.h $(USRBASE)/include/$(APP)
	chmod 755 $(USRBASE)/lib/$(LIBNAME)
	chmod 644 $(USRBASE)/include/$(APP)/*.h
	chmod 644 $(USRBASE)/share/man/man1/$(APP).1.gz
	ln -sf $(USRBASE)/lib/$(LIBNAME) $(USRBASE)/lib/$(SONAME)
	ln -sf $(USRBASE)/lib/$(LIBNAME) $(USRBASE)/lib/$(APP).so
	ldconfig

clean:
	rm -f $(LIBNAME) $(APP) $(APP).so.* $(APP).o $(APP)_tests $(APP)-* *.dot
	rm -f *.dat *.png *.txt *.rb agent.c agent
	rm -f \#* \.#* debian/*.substvars debian/*.log *.so.0.0.1 *.o
	rm -rf deb.* debian/$(APP)0 debian/$(APP)0-dev
	rm ../$(APP)*.deb ../$(APP)*.changes ../$(APP)*.asc ../$(APP)*.dsc ../$(APP)_$(VERSION)*.gz

