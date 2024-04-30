%nproc=4
%mem=5760MB
%chk=meoh_914.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4371 0.1169 0.0112
C -0.0165 -0.0213 0.0092
H 1.7453 -0.7584 -0.3042
H -0.3293 0.8006 -0.6349
H -0.2103 -1.0124 -0.4010
H -0.3612 0.1536 1.0283

