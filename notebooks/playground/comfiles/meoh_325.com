%nproc=4
%mem=5760MB
%chk=meoh_325.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4392 0.0653 0.0522
C -0.0237 -0.0025 0.0006
H 1.7770 -0.1416 -0.8442
H -0.2851 0.4645 -0.9490
H -0.2295 -1.0721 0.0422
H -0.3644 0.5275 0.8900

