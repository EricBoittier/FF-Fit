%nproc=4
%mem=5760MB
%chk=meoh_880.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4276 0.0178 0.0387
C 0.0297 -0.0090 0.0106
H 1.6522 0.6050 -0.7131
H -0.3328 0.3968 -0.9339
H -0.4344 -0.9914 0.0978
H -0.4392 0.5997 0.7837

