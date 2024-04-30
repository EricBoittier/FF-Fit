%nproc=4
%mem=5760MB
%chk=meoh_315.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4145 0.0311 0.0465
C 0.0385 -0.0125 0.0124
H 1.7422 0.3910 -0.8041
H -0.3068 0.3394 -0.9597
H -0.4626 -0.9721 0.1397
H -0.4252 0.6830 0.7119

