%nproc=4
%mem=5760MB
%chk=meoh_968.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4058 0.0124 0.0076
C 0.0339 -0.0260 0.0008
H 1.8407 0.8739 -0.1634
H -0.3059 0.9841 0.2297
H -0.4466 -0.2254 -0.9570
H -0.3470 -0.7358 0.7350

