%nproc=4
%mem=5760MB
%chk=meoh_731.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4209 0.0771 0.0466
C 0.0288 -0.0056 -0.0002
H 1.7547 -0.2690 -0.8074
H -0.3267 -0.8566 -0.5812
H -0.4082 -0.1479 0.9881
H -0.4572 0.9007 -0.3615

