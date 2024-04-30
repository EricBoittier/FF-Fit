%nproc=4
%mem=5760MB
%chk=meoh_620.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4218 0.0191 -0.0504
C 0.0264 -0.0163 0.0026
H 1.7708 0.6440 0.6191
H -0.4436 -0.9926 0.1211
H -0.3013 0.5691 0.8617
H -0.4492 0.4550 -0.8576

