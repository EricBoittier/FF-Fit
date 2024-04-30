%nproc=4
%mem=5760MB
%chk=meoh_135.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4379 0.0041 0.0449
C 0.0096 0.0050 -0.0045
H 1.6663 0.6720 -0.6349
H -0.3461 -1.0018 -0.2234
H -0.4229 0.3160 0.9465
H -0.3755 0.6747 -0.7734

