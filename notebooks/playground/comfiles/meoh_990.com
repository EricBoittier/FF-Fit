%nproc=4
%mem=5760MB
%chk=meoh_990.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4140 0.0631 0.0413
C 0.0297 -0.0040 0.0230
H 1.8484 -0.1102 -0.8200
H -0.5119 0.7831 0.5477
H -0.3283 0.1428 -0.9960
H -0.3474 -0.9845 0.3136

