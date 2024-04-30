%nproc=4
%mem=5760MB
%chk=meoh_511.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4242 0.0510 0.0561
C 0.0395 0.0031 0.0020
H 1.6562 0.0516 -0.8961
H -0.4351 -0.5247 0.8292
H -0.4900 0.9557 -0.0134
H -0.3477 -0.5428 -0.8584

