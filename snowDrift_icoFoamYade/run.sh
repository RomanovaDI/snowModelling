#! /bin/bash
set -x

source PathToOpenFOAM-6/etc/bashrc
./delete.sh
blockMesh
decomposePar
mpiexec -n 1 python scriptYade.py : -n 2 icoFoamYade -parallel
reconstructPar
