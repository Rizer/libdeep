#
# rpm spec for libdeep
#

%define        __spec_install_post %{nil}
%define          debug_package %{nil}
%define        __os_install_post %{_dbpath}/brp-compress

Summary: Genetic programming
Name: libdeep
Version: 1.00
Release: 1
License: BSD
Group: libs
SOURCE0 : %{name}-%{version}.tar.gz
URL: https://launchpad.net/libdeep
Packager: Bob Mottram <bob@sluggish.dyndns.org>
Requires: gnuplot
BuildRoot: %{_tmppath}/%{name}-%{version}-%{release}-root

%description
The aim of libdeep is to make using deep learning easy to include within any C/C++ application.

%prep
%setup -q

%build
 # Empty section.

%install
rm -rf %{buildroot}
mkdir -p  %{buildroot}

# in builddir
cp -a * %{buildroot}

%clean
rm -rf %{buildroot}

%files
%defattr(-,root,root,-)
%config(noreplace) %{_sysconfdir}/%{name}/%{name}.conf
%{_bindir}/*
%attr(644,root,root) /usr/share/man/man1/%{name}.1.gz

%changelog
* Thu Jan 5 2013  Bob Mottram <bob@sluggish.dyndns.org>
- Spec file created

