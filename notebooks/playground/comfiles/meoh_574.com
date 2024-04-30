%nproc=4
%mem=5760MB
%chk=meoh_574.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4134 0.0836 -0.0562
C 0.0249 0.0084 0.0141
H 1.8330 -0.4487 0.6517
H -0.2440 -0.9714 0.4088
H -0.4645 0.7641 0.6285
H -0.3976 0.0134 -0.9906

