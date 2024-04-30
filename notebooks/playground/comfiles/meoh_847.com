%nproc=4
%mem=5760MB
%chk=meoh_847.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4354 0.0059 -0.0442
C -0.0106 -0.0070 0.0158
H 1.7521 0.7815 0.4644
H -0.3062 0.0138 -1.0332
H -0.3436 -0.9199 0.5095
H -0.2997 0.8998 0.5470

