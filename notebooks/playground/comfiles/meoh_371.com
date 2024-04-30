%nproc=4
%mem=5760MB
%chk=meoh_371.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4323 0.0323 -0.0606
C -0.0231 0.0070 0.0070
H 1.8032 0.2560 0.8186
H -0.3656 1.0004 -0.2826
H -0.2338 -0.7743 -0.7232
H -0.2040 -0.2925 1.0394

