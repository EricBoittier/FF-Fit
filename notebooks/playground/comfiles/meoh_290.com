%nproc=4
%mem=5760MB
%chk=meoh_290.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4243 -0.0073 -0.0190
C 0.0233 0.0030 0.0086
H 1.7532 0.8979 0.1625
H -0.3940 -0.0387 -0.9975
H -0.3823 -0.8735 0.5139
H -0.4030 0.8801 0.4954

