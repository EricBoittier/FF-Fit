%nproc=4
%mem=5760MB
%chk=meoh_510.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4254 0.0474 0.0564
C 0.0381 0.0058 0.0012
H 1.6463 0.0905 -0.8975
H -0.4329 -0.5188 0.8326
H -0.4945 0.9565 -0.0213
H -0.3377 -0.5643 -0.8484

