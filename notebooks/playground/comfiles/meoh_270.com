%nproc=4
%mem=5760MB
%chk=meoh_270.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4095 0.0298 -0.0473
C 0.0298 -0.0064 -0.0069
H 1.8621 0.4218 0.7287
H -0.4491 -0.2805 -0.9470
H -0.3323 -0.7232 0.7300
H -0.3494 0.9695 0.2961

