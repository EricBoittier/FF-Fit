%nproc=4
%mem=5760MB
%chk=meoh_235.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4405 0.1075 -0.0409
C -0.0021 -0.0137 0.0074
H 1.6851 -0.6679 0.5062
H -0.3347 -0.6140 -0.8395
H -0.3004 -0.4605 0.9557
H -0.4311 0.9840 -0.0859

