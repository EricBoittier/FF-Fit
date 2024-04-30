%nproc=4
%mem=5760MB
%chk=meoh_240.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4328 0.0998 -0.0450
C -0.0108 -0.0179 0.0050
H 1.7828 -0.5431 0.6066
H -0.3362 -0.5690 -0.8773
H -0.2728 -0.4982 0.9477
H -0.3281 1.0236 -0.0473

