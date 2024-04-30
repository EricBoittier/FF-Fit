%nproc=4
%mem=5760MB
%chk=meoh_366.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4369 0.0498 -0.0644
C -0.0136 0.0172 0.0068
H 1.7472 -0.0642 0.8582
H -0.5035 0.9186 -0.3616
H -0.2308 -0.8378 -0.6334
H -0.1990 -0.2276 1.0526

