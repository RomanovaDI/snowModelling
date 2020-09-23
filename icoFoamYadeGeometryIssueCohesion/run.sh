#! /bin/bash

source /opt/openfoam6/etc/bashrc
./delete.sh
blockMesh
decomposePar
mpiexec -n 1 python scriptYade.py : -n 2 icoFoamYade -parallel
