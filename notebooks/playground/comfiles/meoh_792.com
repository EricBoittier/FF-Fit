%nproc=4
%mem=5760MB
%chk=meoh_792.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4238 0.1077 -0.0325
C 0.0189 -0.0131 -0.0047
H 1.7948 -0.6738 0.4279
H -0.4679 -0.5041 -0.8474
H -0.2370 -0.5572 0.9045
H -0.4549 0.9662 0.0625

