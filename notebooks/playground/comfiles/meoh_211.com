%nproc=4
%mem=5760MB
%chk=meoh_211.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4274 0.1183 0.0097
C 0.0139 -0.0226 0.0035
H 1.7319 -0.7926 -0.1855
H -0.3425 -0.7505 -0.7254
H -0.3808 -0.2263 0.9989
H -0.3732 0.9469 -0.3098

