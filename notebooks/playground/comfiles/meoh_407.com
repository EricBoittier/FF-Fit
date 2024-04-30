%nproc=4
%mem=5760MB
%chk=meoh_407.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4343 0.0414 0.0583
C 0.0188 -0.0011 0.0008
H 1.6152 0.2190 -0.8884
H -0.4521 0.8954 0.4043
H -0.3175 -0.1208 -1.0291
H -0.3778 -0.8531 0.5529

