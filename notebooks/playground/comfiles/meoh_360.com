%nproc=4
%mem=5760MB
%chk=meoh_360.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4263 0.0791 -0.0597
C 0.0373 0.0080 0.0041
H 1.6757 -0.4156 0.7486
H -0.5774 0.8113 -0.4022
H -0.3743 -0.8718 -0.4905
H -0.3481 -0.0907 1.0188

