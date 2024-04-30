%nproc=4
%mem=5760MB
%chk=meoh_950.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4173 0.0252 -0.0541
C 0.0342 -0.0019 0.0075
H 1.7810 0.4866 0.7304
H -0.4555 0.9675 -0.0852
H -0.4205 -0.5635 -0.8085
H -0.3495 -0.4832 0.9070

