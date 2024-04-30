%nproc=4
%mem=5760MB
%chk=meoh_596.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4225 0.0533 -0.0645
C 0.0205 -0.0074 0.0015
H 1.8037 0.1039 0.8370
H -0.3341 -0.9951 0.2962
H -0.3419 0.6710 0.7738
H -0.4917 0.2479 -0.9262

