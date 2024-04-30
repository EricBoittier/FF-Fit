%nproc=4
%mem=5760MB
%chk=meoh_483.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4454 -0.0125 0.0060
C 0.0033 -0.0011 -0.0003
H 1.5698 0.9433 -0.1710
H -0.3838 -0.1273 1.0108
H -0.3120 0.9860 -0.3383
H -0.3971 -0.8049 -0.6181

