%nproc=4
%mem=5760MB
%chk=meoh_619.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4222 0.0199 -0.0525
C 0.0253 -0.0160 0.0043
H 1.7668 0.6265 0.6357
H -0.4333 -0.9969 0.1294
H -0.3103 0.5766 0.8554
H -0.4398 0.4536 -0.8625

