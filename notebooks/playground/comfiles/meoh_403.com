%nproc=4
%mem=5760MB
%chk=meoh_403.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4154 0.0223 0.0427
C 0.0405 0.0127 0.0108
H 1.7482 0.4551 -0.7712
H -0.5129 0.8924 0.3395
H -0.3300 -0.2577 -0.9780
H -0.3951 -0.8104 0.5774

