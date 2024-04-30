%nproc=4
%mem=5760MB
%chk=meoh_382.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4237 0.0012 -0.0375
C 0.0425 -0.0192 0.0051
H 1.6132 0.8399 0.4330
H -0.2829 1.0204 -0.0332
H -0.5254 -0.4867 -0.7993
H -0.4511 -0.3784 0.9081

