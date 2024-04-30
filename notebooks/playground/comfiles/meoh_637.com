%nproc=4
%mem=5760MB
%chk=meoh_637.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4282 -0.0079 -0.0369
C 0.0128 0.0071 0.0184
H 1.7060 0.8686 0.3024
H -0.2883 -1.0393 -0.0314
H -0.4076 0.4536 0.9196
H -0.3727 0.5416 -0.8499

