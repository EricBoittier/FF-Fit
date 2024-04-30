%nproc=4
%mem=5760MB
%chk=meoh_523.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4324 0.0893 0.0399
C -0.0203 -0.0207 0.0051
H 1.8356 -0.3717 -0.7252
H -0.2955 -0.6462 0.8543
H -0.2916 1.0267 0.1371
H -0.2831 -0.3949 -0.9844

