%nproc=4
%mem=5760MB
%chk=meoh_852.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4463 -0.0041 -0.0290
C -0.0182 0.0001 0.0013
H 1.6853 0.8818 0.3152
H -0.3789 0.0472 -1.0263
H -0.3186 -0.9264 0.4907
H -0.2681 0.8464 0.6411

