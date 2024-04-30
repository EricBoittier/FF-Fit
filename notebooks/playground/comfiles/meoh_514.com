%nproc=4
%mem=5760MB
%chk=meoh_514.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4240 0.0619 0.0535
C 0.0293 -0.0067 0.0044
H 1.7067 -0.0618 -0.8767
H -0.4098 -0.5504 0.8409
H -0.4421 0.9761 0.0095
H -0.3477 -0.4818 -0.9013

