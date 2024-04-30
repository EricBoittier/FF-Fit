%nproc=4
%mem=5760MB
%chk=meoh_789.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4315 0.1140 -0.0294
C 0.0024 -0.0184 -0.0018
H 1.7655 -0.7211 0.3598
H -0.4251 -0.5354 -0.8609
H -0.2257 -0.5406 0.9274
H -0.4063 0.9914 0.0369

