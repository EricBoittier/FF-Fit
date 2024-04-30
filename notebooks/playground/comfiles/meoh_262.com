%nproc=4
%mem=5760MB
%chk=meoh_262.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4466 0.0422 -0.0698
C -0.0134 0.0045 0.0116
H 1.6668 0.1436 0.8798
H -0.3591 -0.3809 -0.9477
H -0.2383 -0.7017 0.8108
H -0.4125 1.0007 0.2025

