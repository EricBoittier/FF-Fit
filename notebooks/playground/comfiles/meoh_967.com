%nproc=4
%mem=5760MB
%chk=meoh_967.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4104 0.0097 0.0026
C 0.0275 -0.0236 0.0012
H 1.8212 0.8936 -0.0993
H -0.3023 0.9916 0.2220
H -0.4377 -0.2513 -0.9579
H -0.3373 -0.7218 0.7545

