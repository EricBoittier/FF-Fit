%nproc=4
%mem=5760MB
%chk=meoh_827.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4437 0.0416 -0.0660
C -0.0186 -0.0014 -0.0014
H 1.7137 0.2226 0.8586
H -0.4275 -0.2464 -0.9817
H -0.2113 -0.7888 0.7273
H -0.3108 0.9538 0.4348

