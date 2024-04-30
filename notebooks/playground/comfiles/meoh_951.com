%nproc=4
%mem=5760MB
%chk=meoh_951.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4234 0.0193 -0.0544
C 0.0297 0.0008 0.0082
H 1.7357 0.5451 0.7115
H -0.4615 0.9720 -0.0518
H -0.4104 -0.5556 -0.8193
H -0.3533 -0.4934 0.9010

