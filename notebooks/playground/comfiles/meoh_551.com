%nproc=4
%mem=5760MB
%chk=meoh_551.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4150 0.1124 -0.0156
C 0.0195 -0.0129 0.0079
H 1.8502 -0.7608 0.0763
H -0.3277 -0.8477 0.6168
H -0.3852 0.9039 0.4365
H -0.3696 -0.1411 -1.0022

