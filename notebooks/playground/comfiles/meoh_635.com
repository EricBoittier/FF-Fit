%nproc=4
%mem=5760MB
%chk=meoh_635.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4349 -0.0075 -0.0392
C 0.0009 0.0072 0.0162
H 1.6873 0.8565 0.3485
H -0.2736 -1.0473 -0.0115
H -0.3713 0.4723 0.9290
H -0.3706 0.5369 -0.8610

