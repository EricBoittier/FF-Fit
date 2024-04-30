%nproc=4
%mem=5760MB
%chk=meoh_321.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4226 0.0530 0.0460
C -0.0013 -0.0092 0.0128
H 1.8456 0.0789 -0.8377
H -0.2497 0.4094 -0.9625
H -0.3322 -1.0465 0.0635
H -0.3700 0.6111 0.8297

