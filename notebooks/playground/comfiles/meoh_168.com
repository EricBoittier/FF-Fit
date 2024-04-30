%nproc=4
%mem=5760MB
%chk=meoh_168.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4300 0.0551 0.0549
C 0.0007 -0.0002 0.0017
H 1.8023 -0.0134 -0.8491
H -0.2809 -0.9526 -0.4476
H -0.4048 0.0716 1.0109
H -0.3645 0.8082 -0.6317

