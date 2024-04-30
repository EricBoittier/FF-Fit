%nproc=4
%mem=5760MB
%chk=meoh_162.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4252 0.0412 0.0624
C 0.0272 0.0086 -0.0110
H 1.7316 0.1276 -0.8645
H -0.3257 -0.9574 -0.3722
H -0.4193 0.1033 0.9788
H -0.4739 0.7548 -0.6275

