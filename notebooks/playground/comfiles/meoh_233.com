%nproc=4
%mem=5760MB
%chk=meoh_233.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4372 0.1076 -0.0375
C 0.0093 -0.0082 0.0062
H 1.6814 -0.7028 0.4566
H -0.3415 -0.6355 -0.8133
H -0.3212 -0.4453 0.9484
H -0.4837 0.9586 -0.0949

