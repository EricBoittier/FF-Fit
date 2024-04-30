%nproc=4
%mem=5760MB
%chk=meoh_752.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4143 0.1006 0.0253
C 0.0297 -0.0108 0.0030
H 1.8311 -0.6053 -0.5118
H -0.3525 -0.7533 -0.6975
H -0.3822 -0.2685 0.9787
H -0.4415 0.9436 -0.2318

