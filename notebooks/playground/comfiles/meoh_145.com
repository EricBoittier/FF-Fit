%nproc=4
%mem=5760MB
%chk=meoh_145.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4237 0.0257 0.0447
C 0.0106 -0.0107 0.0089
H 1.7842 0.4874 -0.7411
H -0.3628 -0.9800 -0.3215
H -0.4202 0.2725 0.9692
H -0.2668 0.7247 -0.7463

