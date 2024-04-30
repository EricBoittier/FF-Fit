%nproc=4
%mem=5760MB
%chk=meoh_998.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4239 0.1072 0.0324
C 0.0048 -0.0216 -0.0021
H 1.7870 -0.5897 -0.5533
H -0.3473 0.6790 0.7551
H -0.3800 0.2994 -0.9700
H -0.2592 -1.0480 0.2527

