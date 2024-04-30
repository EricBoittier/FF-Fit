%nproc=4
%mem=5760MB
%chk=meoh_154.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4344 0.0325 0.0609
C -0.0012 -0.0005 -0.0119
H 1.7046 0.3079 -0.8400
H -0.2958 -1.0020 -0.3255
H -0.3560 0.1873 1.0014
H -0.3481 0.7823 -0.6864

