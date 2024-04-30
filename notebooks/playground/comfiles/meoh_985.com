%nproc=4
%mem=5760MB
%chk=meoh_985.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4471 0.0427 0.0539
C -0.0050 0.0025 0.0096
H 1.6197 0.1913 -0.8993
H -0.4802 0.8254 0.5437
H -0.2275 0.0038 -1.0575
H -0.3626 -0.9430 0.4174

