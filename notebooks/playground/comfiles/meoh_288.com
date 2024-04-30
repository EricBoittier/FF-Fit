%nproc=4
%mem=5760MB
%chk=meoh_288.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4349 -0.0098 -0.0272
C 0.0121 0.0037 0.0119
H 1.6734 0.8987 0.2526
H -0.3731 -0.0487 -1.0065
H -0.3748 -0.8626 0.5484
H -0.3865 0.9085 0.4707

