%nproc=4
%mem=5760MB
%chk=meoh_488.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4266 -0.0017 0.0182
C 0.0365 -0.0117 0.0034
H 1.6491 0.8657 -0.3801
H -0.4499 -0.1807 0.9641
H -0.3474 0.9704 -0.2728
H -0.4727 -0.7052 -0.6657

