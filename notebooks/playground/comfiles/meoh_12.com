%nproc=4
%mem=5760MB
%chk=meoh_12.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4331 -0.0108 -0.0003
C 0.0071 0.0020 0.0001
H 1.7220 0.9257 0.0003
H -0.3668 -1.0219 -0.0000
H -0.3593 0.5146 0.8894
H -0.3593 0.5143 -0.8895

