%nproc=4
%mem=5760MB
%chk=meoh_988.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4261 0.0546 0.0468
C 0.0179 -0.0006 0.0199
H 1.7664 0.0103 -0.8712
H -0.5115 0.7980 0.5396
H -0.2868 0.0864 -1.0230
H -0.3584 -0.9690 0.3497

