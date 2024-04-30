%nproc=4
%mem=5760MB
%chk=meoh_693.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4312 0.0248 0.0441
C -0.0004 -0.0084 -0.0014
H 1.7546 0.5143 -0.7409
H -0.3540 -0.9618 -0.3939
H -0.3034 0.1148 1.0384
H -0.3508 0.8232 -0.6128

