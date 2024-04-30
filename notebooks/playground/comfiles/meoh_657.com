%nproc=4
%mem=5760MB
%chk=meoh_657.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4424 -0.0068 0.0044
C -0.0027 -0.0072 -0.0035
H 1.6608 0.9362 -0.1487
H -0.3953 -1.0147 -0.1409
H -0.2895 0.3605 0.9817
H -0.3797 0.6970 -0.7452

