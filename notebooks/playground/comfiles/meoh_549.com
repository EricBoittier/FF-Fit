%nproc=4
%mem=5760MB
%chk=meoh_549.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4197 0.1114 -0.0115
C 0.0052 -0.0105 0.0097
H 1.8483 -0.7697 0.0102
H -0.2787 -0.8541 0.6390
H -0.3876 0.9227 0.4133
H -0.3197 -0.1569 -1.0204

