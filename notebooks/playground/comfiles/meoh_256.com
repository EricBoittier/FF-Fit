%nproc=4
%mem=5760MB
%chk=meoh_256.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4418 0.0554 -0.0722
C 0.0022 0.0040 0.0140
H 1.6489 -0.0621 0.8785
H -0.3613 -0.4301 -0.9174
H -0.2823 -0.6350 0.8500
H -0.4577 0.9840 0.1421

