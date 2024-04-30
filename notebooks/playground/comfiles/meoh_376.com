%nproc=4
%mem=5760MB
%chk=meoh_376.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4177 0.0189 -0.0527
C 0.0205 -0.0093 0.0055
H 1.7603 0.5585 0.6902
H -0.2967 1.0202 -0.1608
H -0.3965 -0.6603 -0.7628
H -0.3558 -0.3226 0.9793

