%nproc=4
%mem=5760MB
%chk=meoh_157.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4313 0.0345 0.0640
C 0.0136 0.0050 -0.0154
H 1.6915 0.2418 -0.8578
H -0.3101 -0.9863 -0.3326
H -0.3810 0.1514 0.9900
H -0.4225 0.7703 -0.6574

