%nproc=4
%mem=5760MB
%chk=meoh_152.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4344 0.0316 0.0574
C -0.0073 -0.0044 -0.0074
H 1.7262 0.3503 -0.8222
H -0.2958 -1.0060 -0.3262
H -0.3509 0.2115 1.0042
H -0.3015 0.7804 -0.7043

