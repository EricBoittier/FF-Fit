%nproc=4
%mem=5760MB
%chk=meoh_759.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4349 0.1133 0.0222
C -0.0025 -0.0165 -0.0081
H 1.7548 -0.7188 -0.3849
H -0.3441 -0.7565 -0.7319
H -0.2831 -0.3226 0.9996
H -0.4140 0.9809 -0.1633

