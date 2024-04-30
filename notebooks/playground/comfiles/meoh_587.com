%nproc=4
%mem=5760MB
%chk=meoh_587.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4422 0.0760 -0.0714
C -0.0132 -0.0237 0.0121
H 1.7078 -0.1411 0.8467
H -0.3027 -1.0077 0.3808
H -0.2775 0.7722 0.7083
H -0.4030 0.2379 -0.9716

