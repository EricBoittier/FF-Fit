%nproc=4
%mem=5760MB
%chk=meoh_332.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4375 0.0850 0.0520
C 0.0153 0.0083 -0.0127
H 1.6389 -0.4923 -0.7140
H -0.4791 0.5173 -0.8401
H -0.2320 -1.0528 -0.0424
H -0.4680 0.3645 0.8971

