%nproc=4
%mem=5760MB
%chk=meoh_618.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4233 0.0204 -0.0549
C 0.0228 -0.0152 0.0063
H 1.7601 0.6090 0.6526
H -0.4180 -1.0035 0.1369
H -0.3184 0.5844 0.8502
H -0.4277 0.4519 -0.8695

