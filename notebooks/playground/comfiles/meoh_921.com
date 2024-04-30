%nproc=4
%mem=5760MB
%chk=meoh_921.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4208 0.1057 -0.0101
C -0.0006 0.0048 0.0028
H 1.8922 -0.7495 0.0736
H -0.4653 0.7988 -0.5819
H -0.2556 -0.9534 -0.4498
H -0.2580 -0.0452 1.0607

