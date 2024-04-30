%nproc=4
%mem=5760MB
%chk=meoh_740.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4406 0.0900 0.0396
C -0.0111 -0.0054 0.0130
H 1.6965 -0.4526 -0.7354
H -0.2070 -0.8144 -0.6908
H -0.3722 -0.2441 1.0133
H -0.3917 0.9310 -0.3950

