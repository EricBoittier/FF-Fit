%nproc=4
%mem=5760MB
%chk=meoh_148.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4289 0.0292 0.0493
C -0.0030 -0.0099 0.0033
H 1.7725 0.4300 -0.7764
H -0.3257 -0.9965 -0.3292
H -0.3781 0.2524 0.9925
H -0.2535 0.7539 -0.7329

