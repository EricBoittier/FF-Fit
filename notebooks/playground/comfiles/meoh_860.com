%nproc=4
%mem=5760MB
%chk=meoh_860.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4244 -0.0096 -0.0052
C 0.0356 0.0008 -0.0074
H 1.6876 0.9340 0.0220
H -0.5159 0.1707 -0.9321
H -0.3556 -0.9311 0.4007
H -0.3900 0.7542 0.6555

