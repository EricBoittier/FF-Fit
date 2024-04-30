%nproc=4
%mem=5760MB
%chk=meoh_336.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4229 0.0944 0.0395
C 0.0395 0.0093 -0.0064
H 1.6832 -0.6391 -0.5562
H -0.5122 0.5395 -0.7827
H -0.2807 -1.0242 -0.1388
H -0.4869 0.3006 0.9025

