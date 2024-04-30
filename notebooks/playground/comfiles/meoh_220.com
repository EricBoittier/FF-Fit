%nproc=4
%mem=5760MB
%chk=meoh_220.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4243 0.1055 -0.0068
C -0.0052 0.0010 -0.0012
H 1.8445 -0.7763 0.0727
H -0.2542 -0.7548 -0.7460
H -0.2565 -0.3543 0.9981
H -0.4199 0.9831 -0.2284

