%nproc=4
%mem=5760MB
%chk=meoh_437.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4470 0.1116 -0.0363
C -0.0135 -0.0053 0.0115
H 1.6187 -0.7709 0.3538
H -0.4180 0.5074 0.8842
H -0.3947 0.4235 -0.9152
H -0.1542 -1.0833 0.0907

