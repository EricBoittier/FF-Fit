%nproc=4
%mem=5760MB
%chk=meoh_811.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4270 0.0788 -0.0715
C 0.0077 -0.0103 0.0272
H 1.7532 -0.2737 0.7828
H -0.3159 -0.3417 -0.9596
H -0.3215 -0.7180 0.7880
H -0.3989 0.9903 0.1743

