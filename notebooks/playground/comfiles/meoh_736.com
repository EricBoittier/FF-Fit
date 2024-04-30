%nproc=4
%mem=5760MB
%chk=meoh_736.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4351 0.0849 0.0441
C 0.0056 -0.0044 0.0058
H 1.6900 -0.3798 -0.7802
H -0.2479 -0.8465 -0.6382
H -0.4012 -0.2054 0.9969
H -0.4279 0.9210 -0.3735

