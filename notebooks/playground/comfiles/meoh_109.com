%nproc=4
%mem=5760MB
%chk=meoh_109.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4466 -0.0145 0.0049
C -0.0068 0.0024 0.0070
H 1.6142 0.9394 -0.1445
H -0.3649 -1.0247 -0.0631
H -0.3817 0.5022 0.9001
H -0.2897 0.5705 -0.8792

