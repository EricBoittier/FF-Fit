%nproc=4
%mem=5760MB
%chk=meoh_13.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4319 -0.0106 -0.0003
C 0.0081 0.0019 0.0001
H 1.7221 0.9256 0.0003
H -0.3668 -1.0216 -0.0000
H -0.3592 0.5145 0.8891
H -0.3593 0.5142 -0.8891

