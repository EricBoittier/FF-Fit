%nproc=4
%mem=5760MB
%chk=meoh_648.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4303 -0.0032 -0.0137
C 0.0061 -0.0065 0.0111
H 1.7448 0.9230 0.0475
H -0.3903 -1.0119 -0.1316
H -0.3227 0.4021 0.9666
H -0.3474 0.5999 -0.8228

