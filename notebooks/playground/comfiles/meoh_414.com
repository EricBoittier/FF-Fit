%nproc=4
%mem=5760MB
%chk=meoh_414.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4460 0.0760 0.0596
C -0.0197 -0.0227 -0.0077
H 1.6415 -0.2081 -0.8578
H -0.2995 0.8870 0.5236
H -0.2815 0.1014 -1.0585
H -0.3188 -0.9319 0.5138

