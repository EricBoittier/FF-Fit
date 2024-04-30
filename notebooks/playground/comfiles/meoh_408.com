%nproc=4
%mem=5760MB
%chk=meoh_408.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4396 0.0467 0.0609
C 0.0099 -0.0054 -0.0019
H 1.5908 0.1564 -0.9012
H -0.4262 0.8997 0.4210
H -0.3085 -0.0828 -1.0414
H -0.3659 -0.8657 0.5519

