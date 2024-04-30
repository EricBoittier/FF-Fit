%nproc=4
%mem=5760MB
%chk=meoh_766.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4358 0.1143 0.0084
C -0.0046 -0.0082 0.0066
H 1.7523 -0.7815 -0.2321
H -0.2974 -0.6917 -0.7905
H -0.2741 -0.4202 0.9791
H -0.4574 0.9627 -0.1944

