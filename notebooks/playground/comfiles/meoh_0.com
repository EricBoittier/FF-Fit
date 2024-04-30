%nproc=4
%mem=5760MB
%chk=meoh_0.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4200 0.0000 0.0000
C 0.0000 0.0000 0.0000
H 1.7467 0.9240 0.0000
H -0.3633 -1.0277 0.0000
H -0.3633 0.5138 0.8900
H -0.3633 0.5138 -0.8900

