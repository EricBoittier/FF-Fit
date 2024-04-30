%nproc=4
%mem=5760MB
%chk=meoh_319.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4156 0.0464 0.0445
C 0.0161 -0.0119 0.0157
H 1.8393 0.1859 -0.8281
H -0.2616 0.3836 -0.9613
H -0.3905 -1.0211 0.0812
H -0.3896 0.6407 0.7887

