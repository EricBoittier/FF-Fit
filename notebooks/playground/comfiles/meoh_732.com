%nproc=4
%mem=5760MB
%chk=meoh_732.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4232 0.0787 0.0463
C 0.0263 -0.0050 0.0006
H 1.7389 -0.2924 -0.8041
H -0.3134 -0.8563 -0.5894
H -0.4130 -0.1590 0.9862
H -0.4568 0.9034 -0.3595

