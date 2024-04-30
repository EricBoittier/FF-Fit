%nproc=4
%mem=5760MB
%chk=meoh_214.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4296 0.1162 0.0046
C -0.0031 -0.0182 0.0024
H 1.7673 -0.7976 -0.1020
H -0.2976 -0.7507 -0.7492
H -0.3133 -0.2656 1.0176
H -0.3525 0.9712 -0.2926

