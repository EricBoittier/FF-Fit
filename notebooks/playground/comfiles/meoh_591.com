%nproc=4
%mem=5760MB
%chk=meoh_591.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4389 0.0677 -0.0689
C -0.0081 -0.0216 0.0054
H 1.7421 -0.0329 0.8576
H -0.3121 -1.0060 0.3614
H -0.2753 0.7393 0.7387
H -0.4391 0.2666 -0.9534

