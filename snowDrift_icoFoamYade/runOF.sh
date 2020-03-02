#! /bin/bash
set -x

source PathToOpenFOAM-6/etc/bashrc
./delete.sh
blockMesh
icoFoam
