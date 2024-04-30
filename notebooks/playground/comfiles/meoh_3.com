%nproc=4
%mem=5760MB
%chk=meoh_3.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4339 -0.0121 -0.0001
C 0.0128 0.0059 0.0001
H 1.7263 0.9233 0.0001
H -0.3639 -1.0170 -0.0001
H -0.3662 0.5120 0.8880
H -0.3661 0.5119 -0.8879

