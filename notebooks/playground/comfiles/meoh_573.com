%nproc=4
%mem=5760MB
%chk=meoh_573.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4165 0.0853 -0.0551
C 0.0187 0.0079 0.0138
H 1.8301 -0.4735 0.6356
H -0.2263 -0.9727 0.4220
H -0.4607 0.7738 0.6235
H -0.3902 0.0092 -0.9966

