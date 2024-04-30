%nproc=4
%mem=5760MB
%chk=meoh_625.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4304 0.0110 -0.0439
C 0.0106 -0.0102 0.0012
H 1.7463 0.7320 0.5399
H -0.4134 -1.0115 0.0769
H -0.2585 0.5381 0.9040
H -0.4457 0.4729 -0.8628

