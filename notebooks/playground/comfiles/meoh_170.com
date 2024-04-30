%nproc=4
%mem=5760MB
%chk=meoh_170.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4329 0.0604 0.0531
C -0.0094 -0.0047 0.0043
H 1.8081 -0.0616 -0.8440
H -0.2673 -0.9517 -0.4699
H -0.3979 0.0654 1.0203
H -0.3167 0.8300 -0.6257

