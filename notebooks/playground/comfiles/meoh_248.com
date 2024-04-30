%nproc=4
%mem=5760MB
%chk=meoh_248.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4092 0.0758 -0.0517
C 0.0296 -0.0060 -0.0001
H 1.8641 -0.2945 0.7334
H -0.3999 -0.5031 -0.8699
H -0.3328 -0.5455 0.8750
H -0.3925 0.9973 0.0581

