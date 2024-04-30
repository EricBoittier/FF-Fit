%nproc=4
%mem=5760MB
%chk=meoh_886.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4279 0.0302 0.0529
C 0.0276 -0.0060 -0.0052
H 1.6862 0.3710 -0.8290
H -0.4418 0.4953 -0.8517
H -0.3543 -1.0253 0.0524
H -0.4245 0.5360 0.8254

