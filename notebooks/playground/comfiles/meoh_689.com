%nproc=4
%mem=5760MB
%chk=meoh_689.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4283 0.0197 0.0424
C 0.0204 -0.0106 -0.0052
H 1.7211 0.5867 -0.7015
H -0.4196 -0.9485 -0.3443
H -0.3617 0.1522 1.0025
H -0.3948 0.8087 -0.5920

