%nproc=4
%mem=5760MB
%chk=meoh_131.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4433 -0.0012 0.0436
C -0.0077 0.0087 -0.0119
H 1.6702 0.7268 -0.5721
H -0.2971 -1.0294 -0.1751
H -0.3563 0.3320 0.9689
H -0.3748 0.6711 -0.7959

