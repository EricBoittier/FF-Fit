%nproc=4
%mem=5760MB
%chk=meoh_616.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4272 0.0209 -0.0599
C 0.0144 -0.0122 0.0113
H 1.7412 0.5733 0.6863
H -0.3758 -1.0206 0.1490
H -0.3319 0.6005 0.8436
H -0.3986 0.4451 -0.8879

