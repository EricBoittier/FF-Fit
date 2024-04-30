%nproc=4
%mem=5760MB
%chk=meoh_476.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4352 -0.0095 -0.0146
C -0.0153 0.0050 0.0039
H 1.7463 0.9042 0.1550
H -0.2641 -0.0698 1.0625
H -0.3416 0.9237 -0.4836
H -0.2800 -0.8808 -0.5736

