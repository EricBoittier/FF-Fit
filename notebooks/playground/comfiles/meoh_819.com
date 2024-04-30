%nproc=4
%mem=5760MB
%chk=meoh_819.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4181 0.0609 -0.0622
C 0.0286 -0.0098 -0.0003
H 1.7975 -0.0303 0.8369
H -0.4960 -0.2612 -0.9221
H -0.2965 -0.7223 0.7579
H -0.3965 0.9501 0.2929

