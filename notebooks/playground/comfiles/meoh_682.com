%nproc=4
%mem=5760MB
%chk=meoh_682.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4344 0.0087 0.0336
C 0.0152 -0.0055 0.0025
H 1.6796 0.6993 -0.6170
H -0.3809 -0.9692 -0.3175
H -0.4098 0.1992 0.9851
H -0.3781 0.7829 -0.6393

