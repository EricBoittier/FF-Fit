%nproc=4
%mem=5760MB
%chk=meoh_494.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4217 0.0138 0.0290
C 0.0123 -0.0135 0.0102
H 1.7974 0.6821 -0.5816
H -0.3628 -0.2899 0.9956
H -0.3263 0.9929 -0.2359
H -0.3614 -0.6589 -0.7848

