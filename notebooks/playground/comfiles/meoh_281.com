%nproc=4
%mem=5760MB
%chk=meoh_281.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4492 -0.0035 -0.0458
C -0.0106 -0.0015 0.0127
H 1.5722 0.7926 0.5124
H -0.3398 -0.0979 -1.0220
H -0.3436 -0.8108 0.6624
H -0.3069 0.9743 0.3976

