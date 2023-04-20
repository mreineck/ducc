# Release guide

These are Philipp's notes for the release process for alpine, nixpkgs and rust.

## Package version overview

Check package versions here:
- https://repology.org/project/python:ducc0/versions


## One-time set-up

For Alpine Linux:
```sh
git clone https://gitlab.alpinelinux.org/alpine/aports
cd aports
git remote add phiadaarr alpine:phiadaarr/aports
git fetch --all
```

For Nix:
```sh
git clone github:NixOS/nixpkgs
cd nixpkgs
git remote add phiadaarr github:phiadaarr/nixpkgs
git fetch --all
```

## Alpine Linux release

```sh
cd aports/community/py3-ducc0
git fetch --all
git checkout master
git pull origin master
git checkout -b py3-ducc0<version>
```
Update version number in `APKBUILD`.
Download the package by hand and run compute checksum (adapt the version numbers):
```sh
wget https://gitlab.mpcdf.mpg.de/mtr/ducc/-/archive/ducc0_0_24_0/ducc-ducc0_0_24_0.tar.gz
sha512sum ducc-ducc0_0_24_0.tar.gz
```
Update checksum in `APKBUILD` and commit:
```sh
git commit -am "community/py3-ducc0: upgrade to <version>"
git push phiadaarr
```
Go to gitlab and open a merge request:

https://gitlab.alpinelinux.org/alpine/aports/-/merge_requests

## nixpkgs release
```sh
git checkout -b python3Packages.ducc<version>
```
Open
```sh
vim pkgs/development/python-modules/ducc0/default.nix
```
and update version number. Change the checksum to something else. Run
```sh
nix-build -A python3Packages.ducc0
```
and get the checksum error. Update the checksum accordingly and run the build
command again. If it compiles, run
```sh
git commit -am "python3Packages.ducc0: <oldversion> -> <newversion>"
git push phiadaarr 
```

Go to github and open a merge request:

https://github.com/NixOS/nixpkgs/pulls

## Rust crates.io

```sh
cd ducc/rust
vim Cargo.toml
```
And update version number. Then:
```sh
cargo publish
```
