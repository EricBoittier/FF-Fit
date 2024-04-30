%nproc=4
%mem=5760MB
%chk=meoh_174.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4347 0.0707 0.0519
C -0.0127 -0.0126 0.0050
H 1.7838 -0.1606 -0.8342
H -0.2730 -0.9425 -0.5006
H -0.4045 0.0534 1.0200
H -0.2684 0.8622 -0.5929

