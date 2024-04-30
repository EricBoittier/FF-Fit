%nproc=4
%mem=5760MB
%chk=meoh_750.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4147 0.0977 0.0263
C 0.0274 -0.0089 0.0085
H 1.8319 -0.5768 -0.5494
H -0.3287 -0.7533 -0.7038
H -0.3894 -0.2657 0.9823
H -0.4364 0.9365 -0.2728

