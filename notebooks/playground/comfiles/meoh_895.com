%nproc=4
%mem=5760MB
%chk=meoh_895.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4358 0.0607 0.0518
C -0.0252 -0.0001 -0.0016
H 1.8157 -0.0483 -0.8450
H -0.3448 0.5570 -0.8824
H -0.2073 -1.0718 -0.0817
H -0.2948 0.3861 0.9814

