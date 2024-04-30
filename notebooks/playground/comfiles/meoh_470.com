%nproc=4
%mem=5760MB
%chk=meoh_470.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4093 0.0047 -0.0345
C 0.0291 0.0051 0.0066
H 1.8550 0.7416 0.4333
H -0.3248 -0.0011 1.0376
H -0.4510 0.8104 -0.5492
H -0.3366 -0.9004 -0.4777

