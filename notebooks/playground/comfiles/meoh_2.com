%nproc=4
%mem=5760MB
%chk=meoh_2.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4498 -0.0160 -0.0001
C 0.0016 0.0074 0.0001
H 1.7248 0.9247 0.0000
H -0.3638 -1.0196 -0.0000
H -0.3679 0.5138 0.8918
H -0.3678 0.5136 -0.8918

