%nproc=4
%mem=5760MB
%chk=meoh_199.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4312 0.0970 0.0297
C -0.0045 0.0013 -0.0007
H 1.8180 -0.6501 -0.4731
H -0.2387 -0.8628 -0.6225
H -0.3552 -0.1757 1.0160
H -0.4286 0.9189 -0.4085

