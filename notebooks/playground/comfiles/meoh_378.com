%nproc=4
%mem=5760MB
%chk=meoh_378.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4154 0.0132 -0.0486
C 0.0375 -0.0148 0.0053
H 1.7146 0.6665 0.6179
H -0.2909 1.0178 -0.1139
H -0.4646 -0.5994 -0.7655
H -0.4140 -0.3343 0.9445

