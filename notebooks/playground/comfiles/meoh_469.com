%nproc=4
%mem=5760MB
%chk=meoh_469.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4086 0.0066 -0.0387
C 0.0343 0.0051 0.0073
H 1.8450 0.7134 0.4814
H -0.3416 0.0132 1.0304
H -0.4625 0.7966 -0.5538
H -0.3494 -0.9045 -0.4548

