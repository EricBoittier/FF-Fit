%nproc=4
%mem=5760MB
%chk=meoh_698.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4321 0.0291 0.0444
C -0.0099 -0.0029 0.0068
H 1.7629 0.4213 -0.7907
H -0.2720 -0.9584 -0.4476
H -0.3160 0.0679 1.0506
H -0.3292 0.8274 -0.6230

