%nproc=4
%mem=5760MB
%chk=meoh_672.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4302 -0.0000 0.0159
C -0.0030 0.0041 0.0183
H 1.7492 0.8052 -0.4427
H -0.2715 -1.0001 -0.3097
H -0.3730 0.2294 1.0184
H -0.3112 0.7018 -0.7604

