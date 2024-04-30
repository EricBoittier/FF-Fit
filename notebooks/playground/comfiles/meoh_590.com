%nproc=4
%mem=5760MB
%chk=meoh_590.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4412 0.0702 -0.0698
C -0.0120 -0.0232 0.0071
H 1.7294 -0.0604 0.8577
H -0.3075 -1.0078 0.3694
H -0.2693 0.7504 0.7306
H -0.4271 0.2645 -0.9588

