%nproc=4
%mem=5760MB
%chk=meoh_821.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4241 0.0563 -0.0617
C 0.0169 -0.0083 -0.0048
H 1.7871 0.0324 0.8483
H -0.4963 -0.2593 -0.9332
H -0.2671 -0.7331 0.7582
H -0.3696 0.9510 0.3393

