%nproc=4
%mem=5760MB
%chk=meoh_615.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4295 0.0210 -0.0624
C 0.0092 -0.0102 0.0139
H 1.7305 0.5548 0.7025
H -0.3513 -1.0294 0.1532
H -0.3373 0.6082 0.8419
H -0.3835 0.4389 -0.8984

