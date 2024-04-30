%nproc=4
%mem=5760MB
%chk=meoh_477.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4400 -0.0114 -0.0116
C -0.0191 0.0048 0.0031
H 1.7105 0.9229 0.1081
H -0.2697 -0.0787 1.0606
H -0.3266 0.9404 -0.4641
H -0.2838 -0.8757 -0.5823

