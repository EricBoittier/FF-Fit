%nproc=4
%mem=5760MB
%chk=meoh_402.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4132 0.0183 0.0383
C 0.0400 0.0145 0.0124
H 1.7780 0.5085 -0.7280
H -0.5129 0.9005 0.3247
H -0.3252 -0.2852 -0.9699
H -0.3883 -0.8026 0.5929

