#!/bin/bash

APP=libdeep
VERSION=1.00
ARCH_TYPE=`uname -m`

if [ $ARCH_TYPE == "x86_64" ]; then
    ARCH_TYPE="amd64"
fi

if [ $ARCH_TYPE == "i686" ]; then
    ARCH_TYPE="i386"
fi
if [ $ARCH_TYPE == "armv5tel" ]; then
    ARCH_TYPE="armel"
fi
# Create a source archive
make clean
make source

# Build the package
fakeroot dpkg-buildpackage -b

# sign files
gpg -ba ../${APP}0-dev_${VERSION}-1_${ARCH_TYPE}.deb
gpg -ba ../${APP}0_${VERSION}-1_${ARCH_TYPE}.deb
gpg -ba ../${APP}_${VERSION}.orig.tar.gz
