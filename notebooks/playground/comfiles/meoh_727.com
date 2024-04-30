%nproc=4
%mem=5760MB
%chk=meoh_727.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4188 0.0717 0.0478
C 0.0224 -0.0094 -0.0014
H 1.8061 -0.1751 -0.8180
H -0.3513 -0.8613 -0.5695
H -0.3585 -0.1087 1.0150
H -0.4253 0.9032 -0.3948

