%nproc=4
%mem=5760MB
%chk=meoh_556.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4174 0.1161 -0.0261
C 0.0330 -0.0198 0.0048
H 1.7964 -0.7480 0.2389
H -0.3829 -0.8498 0.5760
H -0.3883 0.8645 0.4830
H -0.4570 -0.0889 -0.9664

