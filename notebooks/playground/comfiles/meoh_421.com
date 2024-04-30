%nproc=4
%mem=5760MB
%chk=meoh_421.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4083 0.0913 0.0277
C 0.0279 -0.0065 0.0057
H 1.8956 -0.5320 -0.5506
H -0.3763 0.7637 0.6627
H -0.3830 0.1451 -0.9924
H -0.3633 -0.9646 0.3480

