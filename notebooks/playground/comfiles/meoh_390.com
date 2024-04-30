%nproc=4
%mem=5760MB
%chk=meoh_390.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4499 -0.0120 -0.0065
C -0.0236 -0.0057 0.0066
H 1.6349 0.9494 -0.0522
H -0.2765 1.0493 0.1127
H -0.3531 -0.3823 -0.9617
H -0.2795 -0.5736 0.9011

