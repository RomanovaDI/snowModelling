#Deepak Kunhappan, deepak.kunhappan@3sr-grenoble.fr
#Example script of Yade-OpenFOAM coupling.
#get the OpenFOAM solver at : https://github.com/dpkn31/Yade-OpenFOAM-coupling
#get the latest version of Yade with the FoamCoupling engine here : https://gitlab.com/yade-dev/trunk
#Both binary and compiled versions of OpenFOAM-6 can be used. (currently tested for OpenFOAM-6).
#Have MPI (preferably OpenMPI) installed. Note : Both OpenFOAM and Yade has to be compiled with the same MPI version.
#----------------------------------------------------------------------------------------------------------------#
# Usage :
#
# 1. Enable the OpenFOAM coupling engine : (Requires MPI installation)
#       cmake -DCMAKE_INSTALL_PREFIX= /path/to/install  /path/to/sources -DDEBUG=0 -DCHUNKSIZE=1 -DENABLE_MPI=ON
#       make
#       make install
#
# 2. Complile or install OpenFOAM (get binary from here : https://openfoam.org/download/6-ubuntu ) (ubuntu and debian)
#    Example installation for ubuntu (careful, it may be different on your system!):
#       apt install software-properties-common
#       sudo sh -c "wget -O - http://dl.openfoam.org/gpg.key | apt-key add -"
#       add-apt-repository http://dl.openfoam.org/ubuntu
#       apt-get update
#       apt-get install openfoam6
#       . /opt/openfoam6/etc/bashrc
#    Example installation from source (careful, it may be different on your system!):
#       sudo apt install build-essential flex bison git-core cmake zlib1g-dev libboost-system-dev libboost-thread-dev libopenmpi-dev openmpi-bin gnuplot libreadline-dev libncurses-dev libxt-dev libptscotch-dev
#       git clone https://github.com/OpenFOAM/OpenFOAM-6.git
#       source OpenFOAM-6/etc/bashrc
#       cd OpenFOAM-6
#       # export WM_COMPILE_OPTION=Debug
#       ./Allwmake -j 2
#
# 3. Get the Yade-OpenFOAM solver at : https://github.com/dpkn31/Yade-OpenFOAM-coupling
#       git clone https://github.com/dpkn31/Yade-OpenFOAM-coupling.git
#
# 4. Enter the dir : Yade-OpenFOAM-coupling
#     Adapt the file icoFoamYade/icoFoamYade.C on how shear flow velocity (or other parameters) is initialized up to your linking.
#     A commented out example is in icoFoamYade/icoFoamYade.C line 59
#     Compile the solvers and the libs
#       ./Allclean
#       ./Allmake
#
# 5. Once compilation is done, you can run the solver from any dir. Return to this example file directory.
#
# 6. Create a symbolic link to Yade Install
#       ln -s /path/to/yade/install/bin/yade-exec yadeimport.py
#
# 7. Yade side :
#
#    a) In the Yade side, create the scene/simulation similar to this script :
#       (typical yade script but recast in classes. See : https://yade-dev.gitlab.io/trunk/user.html#importing-yade-in-other-python-applications)
#        At present only spheres are supported.  The coupling module is called as "FoamCoupling", see lines 127-130 on how to
#       initialize this.
#
#    b) Set the ids of spheres involved in hydrodynamic interaction:
#        sphereIDs = [b.id for b in O.bodies if type(b.shape)==Sphere]
#        fluidCoupling.setNumParticles(len(sphereIDs))
#        fluidCoupling.setIdList(sphereIDs)
#
#    c) Type of coupling :icoFoamYade is based on simple point force coupling
#      fluidCoupling.isGaussianInterp=False;
#
#
# 8. OpenFOAM side :
#
#      Set up the OpenFOAM in the usual way. (If you're an experienced OpenFOAM user, skip this)
#      I will only highlight the steps to run this example. Note that you can use any type of mesh
#      that is supported by OpenFOAM. (dynamic mesh is not currently supported).
#      The coupling is set in the solver icoFoamYade.C;
#      (nothing has to be modified there, except for velocity initialization.)
#
#       a) create the mesh
#           blockMesh
#
#       b) decompose the mesh:
#           decomposePar
#
#       c) make dir for VTK dump for yade
#           mkdir yadep
#
#       d) run the example :
#           mpiexec -n 1 python3 scriptYade.py : -n 2 icoFoamYade -parallel
#
# 9. Notes (OpenFOAM side):
#     to configure the mesh, edit :  system/blockMeshDict
#     to change the number of subdomains, edit : system/decomposeParDict
#     to change solver settings, edit:   system/controlDict
#     to change solution settings, edit : system/fvSolution (linear solver settings and choice of linear solvers)
#     to change discretization schemes, edit : system/fvSceme (for gradient and divergence calculation schemes)
#     to change fluid properties and particle density , edit : constant/transportProperties
#     to set BCs : edit the files in 0 for each field variables.
#
#
#10. Post-Processing : Paraview or ParaFOAM can be used to visualize the results, you can also use the OpenFOAM
#    utilities to postprocess the fluid side.
#
#       DISCLAIMER : The settings provided in this example are not universal, depending on your problem, you
#       should change the solver settings such as timestep, under relaxation factors, linear solver settings etc.
#----------------------------------------------------------------------------------------------------------------#


from __future__ import print_function
import sys
from yadeimport import *
from yade.utils import *

initMPI()                           #Initialize the mpi environment, always required.
fluidCoupling = yade.FoamCoupling();     #Initialize the engine
fluidCoupling.getRank();            #part of Initialization.


#example of spheres in shear flow : two-way point force coupling
class simulation():

	def __init__(self):
		epsilon = 1e-08
		minval = epsilon
		maxval = 1-epsilon
		length = maxval - minval
		halflength = length / 2
		radius = 0.01
		radiusEpsilon = 0.5

		O.periodic = True
		O.cell.setBox(maxval, maxval, maxval)

		numspheres=1000
		young = 1#ice#5e6#1
		poisson = 0.2#0.4
		density = 20

		O.materials.append(FrictMat(young=young,poisson=poisson,frictionAngle=radians(85),density=density,label='spheremat'))
		O.materials.append(FrictMat(young=4e10,poisson=0.35,frictionAngle=radians(70),density=1500,label='wallmat'))

		#wall coords, use facets for wall BC:
		v0 = Vector3(minval, minval, minval)
		v1 = Vector3(maxval, minval, minval)
		v2 = Vector3(maxval, maxval, minval)
		v3 = Vector3(maxval, minval, minval)

		v4 = Vector3(minval,minval, maxval)
		v5 = Vector3(maxval,minval, maxval)
		v6 = Vector3(maxval,maxval, maxval)
		v7 = Vector3(minval,maxval, maxval)

		ff0 = facet([v0,v1,v5], fixed=True, wire=True, material='wallmat')
		ff1 = facet([v0,v5,v4], fixed=True, wire=True, material='wallmat')
		bf0 = facet([v3,v7,v6], fixed=True, wire=True, material='wallmat')
		bf1 = facet([v3,v6,v2], fixed=True, wire=True, material='wallmat')
		lf0 = facet([v0,v4,v3], fixed=True, wire=False, material='wallmat')
		lf1 = facet([v0,v7,v3], fixed=True, wire=False, material='wallmat')
		rf0 = facet([v1,v2,v6], fixed=True, wire=False, material='wallmat')
		rf1 = facet([v1,v6,v5], fixed=True, wire=False, material='wallmat')
		uf0 = facet([v4,v5,v6], fixed=True, wire=False, material='wallmat')
		uf1 = facet([v4,v6,v7], fixed=True, wire=False, material='wallmat')
		df0 = facet([v0,v3,v2], fixed=True, wire=False, material='wallmat')
		df1 = facet([v0,v2,v1], fixed=True, wire=False, material='wallmat')
		O.bodies.append([ff0, ff1, bf0, bf1, lf0, lf1, rf0, rf1, uf0, uf1, df0, df1])

		#spheres
		mn, mx= Vector3(minval+2*radius, minval+2*radius, minval+2*radius), Vector3(maxval-2*radius, maxval-2*radius, maxval-2*radius)

		sp = pack.SpherePack()
		sp.makeCloud(mn,mx,rMean=radius,rRelFuzz=radiusEpsilon, num=numspheres)
		O.bodies.append([sphere(center,rad,material='spheremat') for center,rad in sp])

		sphereIDs = [b.id for b in O.bodies if type(b.shape)==Sphere]

		#coupling engine settings
		fluidCoupling.setNumParticles(len(sphereIDs))
		fluidCoupling.setIdList(sphereIDs)
		fluidCoupling.isGaussianInterp=False  #use pimpleFoamYade for gaussianInterp

		O.dt=1e-4
		#O.dynDt=False

		O.engines=[
			ForceResetter(),
			InsertionSortCollider([Bo1_Sphere_Aabb(), Bo1_Facet_Aabb(), Bo1_Wall_Aabb()], allowBiggerThanPeriod=True),
			InteractionLoop(
				[Ig2_Sphere_Sphere_ScGeom(),Ig2_Facet_Sphere_ScGeom()],
				[Ip2_FrictMat_FrictMat_FrictPhys()],
				[Law2_ScGeom_FrictPhys_CundallStrack()]
			),
			#GlobalStiffnessTimeStepper(timestepSafetyCoefficient=0.5, label = "ts"),
			fluidCoupling, #to be called after timestepper
			PyRunner(command='sim.printMessage()', iterPeriod= 1, label='outputMessage'),
			NewtonIntegrator(damping=0.0, gravity = (0.0, 0.0, -9.81)),# add small damping in case of stability issues.. ~ 0.1 max, also note : If gravity is needed, set it in constant/g dir
			VTKRecorder(fileName='yadep/3d-vtk-',recorders=['spheres'],virtPeriod=0.02)#iterPeriod=1000)
		]

	def printMessage(self):
		print("********************************YADE-ITER = " + str(O.iter) +" **********************************")
#		if O.iter == 4000:
#			maxVel = 0.05
#			for b in O.bodies:
#				if type(b.shape)==Sphere:
#					bodyVel = abs(b.state.vel.norm())
#					if bodyVel > maxVel:
#						raise ValueError("Body velocity exceeds imposed shear velocity by ", abs(bodyVel-maxVel))

	def irun(self,num):
		O.run(num,1)

if __name__=="__main__":
	sim = simulation()
	sim.irun(10000)
	fluidCoupling.killMPI()

import builtins
builtins.sim=sim
