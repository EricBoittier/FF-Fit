%nproc=4
%mem=5760MB
%chk=meoh_486.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4344 -0.0069 0.0138
C 0.0272 -0.0076 0.0012
H 1.5992 0.9067 -0.3006
H -0.4382 -0.1562 0.9756
H -0.3364 0.9769 -0.2929
H -0.4582 -0.7430 -0.6404

