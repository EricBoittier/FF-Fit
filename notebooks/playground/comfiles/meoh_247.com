%nproc=4
%mem=5760MB
%chk=meoh_247.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4094 0.0786 -0.0497
C 0.0259 -0.0078 -0.0006
H 1.8749 -0.3231 0.7135
H -0.3946 -0.5127 -0.8704
H -0.3261 -0.5393 0.8834
H -0.3750 1.0048 0.0450

