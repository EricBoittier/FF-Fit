%nproc=4
%mem=5760MB
%chk=meoh_428.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4241 0.1086 -0.0040
C 0.0191 0.0037 0.0145
H 1.7732 -0.7942 -0.1578
H -0.4691 0.6161 0.7726
H -0.4107 0.2252 -0.9623
H -0.2660 -1.0294 0.2135

