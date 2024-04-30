%nproc=4
%mem=5760MB
%chk=meoh_954.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4446 0.0040 -0.0530
C 0.0073 0.0074 0.0093
H 1.5986 0.6977 0.6220
H -0.4483 0.9968 0.0511
H -0.3635 -0.5363 -0.8596
H -0.3451 -0.5257 0.8923

