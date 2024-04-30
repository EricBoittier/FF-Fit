%nproc=4
%mem=5760MB
%chk=meoh_588.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4429 0.0743 -0.0711
C -0.0146 -0.0243 0.0105
H 1.7114 -0.1146 0.8523
H -0.3026 -1.0089 0.3790
H -0.2702 0.7671 0.7152
H -0.4086 0.2503 -0.9680

