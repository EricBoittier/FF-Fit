%nproc=4
%mem=5760MB
%chk=meoh_751.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4140 0.0991 0.0257
C 0.0294 -0.0098 0.0058
H 1.8334 -0.5905 -0.5303
H -0.3421 -0.7529 -0.6999
H -0.3874 -0.2666 0.9797
H -0.4401 0.9396 -0.2517

