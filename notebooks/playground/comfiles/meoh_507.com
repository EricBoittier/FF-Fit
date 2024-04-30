%nproc=4
%mem=5760MB
%chk=meoh_507.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4312 0.0375 0.0554
C 0.0210 0.0108 -0.0001
H 1.6454 0.2087 -0.8855
H -0.3960 -0.5050 0.8649
H -0.4727 0.9812 -0.0509
H -0.2851 -0.6237 -0.8319

