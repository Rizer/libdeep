APP=libdeep
VERSION=1.00
SOURCEDIR=.
ARCH_TYPE=`uname -m`
RELEASE=1
SONAME=${APP}.so.0
LIBNAME=${APP}-${VERSION}.so.0.0.${RELEASE}
USRBASE=/usr

sudo yum groupinstall "Development Tools"
sudo yum install rpmdevtools

make

rm -rf ~/rpmbuild
sudo rm -rf rpmpackage/$APP-$VERSION
rpmdev-setuptree
mkdir rpmpackage/$APP-$VERSION

mkdir rpmpackage/$APP-$VERSION/etc
mkdir rpmpackage/$APP-$VERSION/etc/$APP
mkdir rpmpackage/$APP-$VERSION/usr
mkdir rpmpackage/$APP-$VERSION/usr/lib
mkdir rpmpackage/$APP-$VERSION/usr/include
mkdir rpmpackage/$APP-$VERSION/usr/include/$APP
mkdir rpmpackage/$APP-$VERSION/usr/bin
mkdir rpmpackage/$APP-$VERSION/usr/share
mkdir rpmpackage/$APP-$VERSION/usr/share/man
mkdir rpmpackage/$APP-$VERSION/usr/share/man/man1
install -m 755 $LIBNAME rpmpackage/$APP-$VERSION/usr/lib
install -m 644 src/*.h rpmpackage/$APP-$VERSION/usr/include/$APP
install -m 644 man/$APP.1.gz rpmpackage/$APP-$VERSION/usr/share/man/man1
#ln -sf /usr/lib/$LIBNAME /usr/lib/$SONAME
#ln -sf /usr/lib/$LIBNAME /usr/lib/$APP.so

cd rpmpackage
mkdir $APP-$VERSION/etc/$APP
install -m 644 $APP.conf $APP-$VERSION/etc/$APP/
tar -zcvf $APP-$VERSION.tar.gz $APP-$VERSION/

rm -rf ~/rpmbuild/BUILD/$APP-$VERSION
rm ~/rpmbuild/SOURCES/$APP*.*
cp $APP-$VERSION.tar.gz ~/rpmbuild/SOURCES/
cp $APP.spec ~/rpmbuild/SPECS/

rpmbuild -ba ~/rpmbuild/SPECS/$APP.spec

sudo rm -rf $APP-$VERSION
rm $APP-$VERSION.tar.gz
cp -r ~/rpmbuild/RPMS/* .
cd ..
echo ---------------------------------------------------------
echo RPM files can be found in the rpmpackage directory
echo under architecture subdirectories.
echo ---------------------------------------------------------
