%nproc=4
%mem=5760MB
%chk=meoh_224.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4196 0.1017 -0.0156
C 0.0159 0.0095 -0.0019
H 1.8299 -0.7636 0.1931
H -0.2878 -0.7350 -0.7378
H -0.2953 -0.3931 0.9620
H -0.5097 0.9494 -0.1710

