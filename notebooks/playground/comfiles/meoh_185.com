%nproc=4
%mem=5760MB
%chk=meoh_185.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4205 0.0847 0.0520
C 0.0330 -0.0066 -0.0060
H 1.7117 -0.4142 -0.7397
H -0.3398 -0.8807 -0.5398
H -0.4172 -0.0372 0.9862
H -0.4363 0.8516 -0.4870

