%nproc=4
%mem=5760MB
%chk=meoh_172.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4347 0.0657 0.0521
C -0.0145 -0.0090 0.0053
H 1.8013 -0.1108 -0.8395
H -0.2640 -0.9485 -0.4877
H -0.3974 0.0598 1.0235
H -0.2822 0.8483 -0.6123

