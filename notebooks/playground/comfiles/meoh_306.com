%nproc=4
%mem=5760MB
%chk=meoh_306.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4492 -0.0013 0.0447
C -0.0096 0.0033 -0.0107
H 1.5688 0.7414 -0.5835
H -0.3199 0.2254 -1.0318
H -0.3245 -0.9805 0.3373
H -0.3540 0.7803 0.6717

