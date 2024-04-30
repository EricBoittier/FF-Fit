%nproc=4
%mem=5760MB
%chk=meoh_859.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4272 -0.0100 -0.0076
C 0.0317 0.0015 -0.0083
H 1.6789 0.9347 0.0609
H -0.5174 0.1539 -0.9375
H -0.3541 -0.9257 0.4153
H -0.3790 0.7617 0.6562

