%nproc=4
%mem=5760MB
%chk=meoh_246.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4108 0.0816 -0.0481
C 0.0210 -0.0095 -0.0007
H 1.8789 -0.3526 0.6954
H -0.3872 -0.5219 -0.8719
H -0.3174 -0.5338 0.8930
H -0.3582 1.0119 0.0314

