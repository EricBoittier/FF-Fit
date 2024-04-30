%nproc=4
%mem=5760MB
%chk=meoh_845.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4266 0.0098 -0.0490
C 0.0048 -0.0089 0.0206
H 1.7731 0.7313 0.5165
H -0.3102 0.0055 -1.0228
H -0.3651 -0.9076 0.5142
H -0.3408 0.9074 0.4991

