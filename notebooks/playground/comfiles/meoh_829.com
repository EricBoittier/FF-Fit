%nproc=4
%mem=5760MB
%chk=meoh_829.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4442 0.0366 -0.0675
C -0.0174 0.0003 0.0040
H 1.6955 0.2854 0.8465
H -0.4071 -0.2262 -0.9884
H -0.2235 -0.8114 0.7017
H -0.3226 0.9520 0.4391

