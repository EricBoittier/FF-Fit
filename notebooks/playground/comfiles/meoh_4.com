%nproc=4
%mem=5760MB
%chk=meoh_4.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4383 -0.0095 -0.0002
C -0.0046 0.0010 0.0001
H 1.7282 0.9267 0.0001
H -0.3657 -1.0275 -0.0001
H -0.3597 0.5167 0.8923
H -0.3597 0.5165 -0.8923

