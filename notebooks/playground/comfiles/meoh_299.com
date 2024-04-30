%nproc=4
%mem=5760MB
%chk=meoh_299.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4275 -0.0033 0.0199
C 0.0002 0.0040 -0.0074
H 1.8053 0.8527 -0.2719
H -0.3621 0.0733 -1.0331
H -0.2969 -0.9524 0.4229
H -0.3482 0.8162 0.6305

